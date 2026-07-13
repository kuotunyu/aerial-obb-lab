"""DOTA8 end-to-end smoke test on the local GPU (RTX 2070 8GB).

Validates the whole pipeline at tiny scale before any Colab spend:
  1. ENV      - torch CUDA available, correct ultralytics version
  2. PREDICT  - pretrained yolo26n-obb produces rotated boxes on a sample image
  3. TRAIN    - short fine-tune on dota8.yaml, interrupted mid-run on purpose
  4. RESUME   - training resumes from last.pt and finishes all epochs
  5. VAL      - validation metrics come out sane
  6. HF_PUSH  - checkpoint callback pushed last.pt to a private HF test repo
               (SKIP if not logged in to Hugging Face)
  7. EXPORT   - ONNX export + inference through YOLO("*.onnx") yields OBBs

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EPOCHS = 4
INTERRUPT_AFTER = 2  # epochs completed before the simulated disconnect
IMGSZ = 1024
BATCH = 4
RUNS = ROOT / "runs"
SAMPLE_IMAGE = "https://ultralytics.com/images/boats.jpg"

results: list[tuple[str, str, str]] = []  # (step, status, detail)


def record(step: str, status: str, detail: str = "") -> None:
    results.append((step, status, detail))
    print(f"[smoke] {step}: {status}  {detail}")


class SimulatedDisconnect(Exception):
    pass


def main() -> int:
    # ---- 1. ENV ----------------------------------------------------------
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO

        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        gpu = torch.cuda.get_device_name(0)
        # exercise an actual CUDA op (driver/arch mismatches only surface at runtime)
        x = (torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")).sum().item()
        assert x == 64 * 64 * 64
        record("ENV", "OK", f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, {gpu}")
    except Exception as e:
        record("ENV", "FAIL", str(e))
        return summary()

    device = 0

    # ---- 2. PREDICT (pretrained) ----------------------------------------
    try:
        model = YOLO("yolo26n-obb.pt")
        pred = model.predict(SAMPLE_IMAGE, imgsz=IMGSZ, device=device, verbose=False)
        n_boxes = len(pred[0].obb) if pred[0].obb is not None else 0
        assert n_boxes > 0, "no OBB detections on sample image"
        pred[0].save(str(RUNS / "smoke_predict_pretrained.jpg"))
        record("PREDICT", "OK", f"{n_boxes} rotated boxes on boats.jpg")
    except Exception as e:
        record("PREDICT", "FAIL", str(e))
        return summary()

    # ---- HF setup (used by TRAIN callback; optional) ---------------------
    hf_repo = None
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami()["name"]
        from obbkit.hf_checkpoint import ensure_repo, install_hf_checkpoint_callback

        hf_repo = ensure_repo(f"{user}/yolo26-dota-obb-smoke", private=True)
        print(f"[smoke] HF logged in as {user}, test repo: {hf_repo}")
    except Exception as e:
        print(f"[smoke] HF not available ({e}); HF_PUSH will be SKIP")

    # ---- 3. TRAIN with simulated disconnect ------------------------------
    train_name = "smoke_dota8"
    try:
        model = YOLO("yolo26n-obb.pt")
        if hf_repo:
            install_hf_checkpoint_callback(model, hf_repo, every_n_epochs=1)

        def kill_switch(trainer):
            if trainer.epoch + 1 >= INTERRUPT_AFTER:
                raise SimulatedDisconnect(f"simulated disconnect after epoch {trainer.epoch + 1}")

        model.add_callback("on_fit_epoch_end", kill_switch)
        try:
            model.train(
                data="dota8.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                device=device, workers=2, project=str(RUNS), name=train_name, exist_ok=True,
            )
            record("TRAIN", "FAIL", "kill switch never fired - interrupt logic broken")
        except SimulatedDisconnect as e:
            last = RUNS / train_name / "weights" / "last.pt"
            assert last.is_file(), "last.pt missing after interrupted run"
            record("TRAIN", "OK", f"interrupted as planned ({e}); last.pt saved")
    except Exception as e:
        traceback.print_exc()
        record("TRAIN", "FAIL", str(e))
        return summary()

    # ---- 4. RESUME --------------------------------------------------------
    try:
        last = RUNS / train_name / "weights" / "last.pt"
        model = YOLO(str(last))
        if hf_repo:
            install_hf_checkpoint_callback(model, hf_repo, every_n_epochs=1)
        model.train(resume=True)
        import csv

        with open(RUNS / train_name / "results.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        last_epoch = int(float(rows[-1]["epoch"])) if rows else 0
        assert last_epoch == EPOCHS, f"expected final epoch {EPOCHS} in results.csv, got {last_epoch}"
        record("RESUME", "OK", f"resumed and finished all {EPOCHS} epochs")
    except Exception as e:
        traceback.print_exc()
        record("RESUME", "FAIL", str(e))
        return summary()

    # ---- 5. VAL ------------------------------------------------------------
    try:
        best = RUNS / train_name / "weights" / "best.pt"
        metrics = YOLO(str(best)).val(data="dota8.yaml", imgsz=IMGSZ, device=device, verbose=False)
        rd = {k: round(float(v), 4) for k, v in metrics.results_dict.items()}
        record("VAL", "OK", f"{rd}")
    except Exception as e:
        record("VAL", "FAIL", str(e))

    # ---- 6. HF_PUSH check ---------------------------------------------------
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            assert api.file_exists(hf_repo, "checkpoints/last.pt"), "checkpoints/last.pt not on Hub"
            assert api.file_exists(hf_repo, "checkpoints/best.pt"), "checkpoints/best.pt not on Hub"
            record("HF_PUSH", "OK", f"last/best/results on {hf_repo}")
        except Exception as e:
            record("HF_PUSH", "FAIL", str(e))
    else:
        record("HF_PUSH", "SKIP", "not logged in to Hugging Face (run: uv run hf auth login)")

    # ---- 7. EXPORT ONNX + ORT inference -------------------------------------
    try:
        best = RUNS / train_name / "weights" / "best.pt"
        # simplify=False: onnxslim 0.1.94 hard-crashes (0xC0000005) on this Windows box;
        # Colab notebooks keep the default (slimming works on Linux).
        # device="cpu": keeps ultralytics on the pinned CPU onnxruntime instead of
        # auto-installing onnxruntime-gpu (whose current wheels fail DLL init here);
        # matches how the HF Space CPU demo will run.
        # opset=19: default (22) exceeds what the pinned ORT 1.20 loads, and torch 2.8's
        # exporter only truly supports <=20 anyway
        onnx_path = YOLO(str(best)).export(format="onnx", imgsz=IMGSZ, simplify=False, opset=19)
        # task="obb" is required: without it the ONNX gets treated as plain detect and
        # results[0].obb stays None
        pred = YOLO(onnx_path, task="obb").predict(SAMPLE_IMAGE, imgsz=IMGSZ, device="cpu", verbose=False)
        assert pred[0].obb is not None, "exported model lost the OBB head/task"
        n_boxes = len(pred[0].obb)
        record("EXPORT", "OK", f"ONNX exported ({Path(onnx_path).name}), ORT inference ran, {n_boxes} boxes")
    except Exception as e:
        traceback.print_exc()
        record("EXPORT", "FAIL", str(e))

    return summary()


def summary() -> int:
    print("\n" + "=" * 62)
    print(f"{'STEP':<10}{'STATUS':<8}DETAIL")
    print("-" * 62)
    for step, status, detail in results:
        print(f"{step:<10}{status:<8}{detail[:80]}")
    print("=" * 62)
    failed = [s for s, st, _ in results if st == "FAIL"]
    print("SMOKE TEST:", "ALL GREEN" if not failed else f"FAILED -> {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
