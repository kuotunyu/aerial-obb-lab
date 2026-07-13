"""HF Hub checkpoint push + resume helpers for Ultralytics training runs.

Designed for Colab runs that may disconnect: every N epochs the trainer's
last.pt / best.pt / results.csv are pushed to a Hugging Face model repo, and a
fresh run can pull last.pt back and continue with resume=True.

Used by scripts/smoke_test.py (local dry-run) and inlined into
notebooks/01_train_dotav1_a100.ipynb (the notebook is self-contained).
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

CKPT_DIR = "checkpoints"


def ensure_repo(repo_id: str, private: bool = True, token: str | None = None) -> str:
    """Create the model repo if needed and return its repo_id."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return repo_id


def install_hf_checkpoint_callback(
    model,
    repo_id: str,
    every_n_epochs: int = 2,
    token: str | None = None,
) -> None:
    """Register callbacks on a YOLO model that push checkpoints to HF Hub.

    - on_model_save: every `every_n_epochs` epochs upload last.pt (+ results.csv)
    - on_train_end: upload best.pt, last.pt and results.csv one final time
    """
    api = HfApi(token=token)

    def _upload(path: Path, dest: str) -> None:
        if path and Path(path).is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type="model",
            )

    def on_model_save(trainer) -> None:
        # trainer.epoch is 0-indexed and on_model_save fires after the ckpt is written
        if (trainer.epoch + 1) % every_n_epochs != 0:
            return
        try:
            _upload(trainer.last, f"{CKPT_DIR}/last.pt")
            _upload(Path(trainer.save_dir) / "results.csv", f"{CKPT_DIR}/results.csv")
            print(f"[hf_checkpoint] pushed last.pt @ epoch {trainer.epoch + 1} -> {repo_id}")
        except Exception as e:  # never let an upload hiccup kill a long training run
            print(f"[hf_checkpoint] WARNING: push failed @ epoch {trainer.epoch + 1}: {e}")

    def on_train_end(trainer) -> None:
        try:
            _upload(trainer.best, f"{CKPT_DIR}/best.pt")
            _upload(trainer.last, f"{CKPT_DIR}/last.pt")
            _upload(Path(trainer.save_dir) / "results.csv", f"{CKPT_DIR}/results.csv")
            print(f"[hf_checkpoint] final push (best/last/results) -> {repo_id}")
        except Exception as e:
            print(f"[hf_checkpoint] WARNING: final push failed: {e}")

    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_train_end", on_train_end)


def pull_resume_checkpoint(repo_id: str, dest_dir: str | Path, token: str | None = None) -> Path | None:
    """Download checkpoints/last.pt from the repo if it exists; return local path or None."""
    api = HfApi(token=token)
    remote = f"{CKPT_DIR}/last.pt"
    if not (api.repo_exists(repo_id) and api.file_exists(repo_id, remote)):
        return None
    local = hf_hub_download(repo_id=repo_id, filename=remote, local_dir=str(dest_dir), token=token)
    return Path(local)
