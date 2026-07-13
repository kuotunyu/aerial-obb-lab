# %% [markdown]
# # YOLO26m-OBB × DOTAv1 — Fine-tuning on Colab (A100)
#
# **這個 notebook 是自包含的**：不需要 clone 任何 repo，直接「全部執行」即可。
#
# ## 使用步驟
# 1. 執行階段 → 變更執行階段類型 → **A100 GPU**（搶不到就改用 T4，並把下方 `PRESET` 改成 `"T4"`）
# 2. 左側 🔑 **Secrets**：新增 `HF_TOKEN`（Hugging Face **write** token），並開啟「筆記本存取權」
# 3. 執行階段 → **全部執行**，掛機等完成（A100 多尺度約 4–6 小時；`SPLIT_RATES=[1.0]` 約可減半）
# 4. 跑完把最後一格的 `=== PASTE BACK ===` 區塊貼回 Claude Code session
# 5. **記得**：執行階段 → 中斷連線並刪除執行階段
#
# ## 斷線怎麼辦？
# 訓練中每 2 個 epoch 會把 `last.pt` push 到你的 HF model repo。
# **重新開啟本 notebook 再「全部執行」一次即可**：它會自動偵測 HF 上的 checkpoint 並 `resume=True` 接續訓練。

# %% [markdown]
# ## 0. 設定

# %%
PRESET = "A100"  # "A100" 或 "T4"

MODEL = "yolo26m-obb.pt"   # DOTAv1 官方預訓練 OBB 權重（fine-tune 起點）
EPOCHS = 60
IMGSZ = 1024
PATIENCE = 15              # 早停：val 指標 15 epochs 無進步就停
PUSH_EVERY = 2             # 每 N epochs push checkpoint 到 HF

# split_dota 官方多尺度切圖設定；改成 [1.0] 可讓資料量與訓練時間約減半
SPLIT_RATES = [0.5, 1.0, 1.5]
SPLIT_GAP = 500

RUN_NAME = "yolo26m-obb-dotav1"
HF_MODEL_REPO_NAME = "yolo26m-obb-dota"        # -> {你的帳號}/yolo26m-obb-dota（訓練期間 private）
HF_DATASET_CACHE_NAME = "dotav1-split-cache"   # 切好的 tiles 快取（private，遵守 DOTA 學術授權）
CACHE_MAX_GB = 30                              # tar 超過此大小就不上傳快取

PRESETS = {
    "A100": dict(batch=16, workers=8),
    "T4": dict(batch=4, workers=2),
}

# %% [markdown]
# ## 1. 安裝與登入

# %%
# %pip install -q ultralytics==8.4.93

# %%
from google.colab import userdata
from huggingface_hub import HfApi, login

HF_TOKEN = userdata.get("HF_TOKEN")
login(HF_TOKEN)
HF_USER = HfApi().whoami()["name"]
print("Hugging Face user:", HF_USER)

MODEL_REPO = f"{HF_USER}/{HF_MODEL_REPO_NAME}"
CACHE_REPO = f"{HF_USER}/{HF_DATASET_CACHE_NAME}"

# %%
import torch

assert torch.cuda.is_available(), "沒有 GPU！請確認執行階段類型"
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.0f} GB)")
if "T4" in gpu and PRESET == "A100":
    print("⚠️ 偵測到 T4 但 PRESET=A100 —— 自動改用 T4 參數")
    PRESET = "T4"

# %% [markdown]
# ## 2. 資料準備：DOTAv1 下載 → split_dota 切圖（含 HF 快取）
#
# ultralytics 的 DOTAv1（2GB）是**原始大圖**，必須先切成 1024×1024 重疊 tiles。
# 切圖結果會打包快取到 HF 私有 dataset repo，斷線重跑先拉快取、免重切。

# %%
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

api = HfApi()
DATASETS = Path("/content/datasets")

from ultralytics import settings

settings.update({"datasets_dir": str(DATASETS)})  # 明確固定，避免預設值推斷不一致
SPLIT_DIR = DATASETS / "DOTAv1-split"
DATA_YAML = DATASETS / "DOTAv1-split.yaml"
rate_tag = "-".join(str(r) for r in SPLIT_RATES).replace(".", "")
CACHE_FILE = f"dotav1-split-r{rate_tag}-g{SPLIT_GAP}.tar"

def split_ready() -> bool:
    return (SPLIT_DIR / "images" / "train").is_dir() and (SPLIT_DIR / "labels" / "val").is_dir()

if split_ready():
    print("split dataset already on disk")
elif api.repo_exists(CACHE_REPO, repo_type="dataset") and api.file_exists(
    CACHE_REPO, CACHE_FILE, repo_type="dataset"
):
    print("pulling split cache from HF ...")
    tar_path = hf_hub_download(CACHE_REPO, CACHE_FILE, repo_type="dataset", local_dir="/content/cache")
    DATASETS.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(DATASETS)
    print("cache extracted ->", SPLIT_DIR)
else:
    from ultralytics.data.split_dota import split_trainval
    from ultralytics.data.utils import check_det_dataset

    print("downloading DOTAv1 (2GB, original large images) ...")
    check_det_dataset("DOTAv1.yaml")  # auto-download to /content/datasets/DOTAv1
    print("splitting into 1024 tiles (rates =", SPLIT_RATES, ") ...")
    split_trainval(
        data_root=str(DATASETS / "DOTAv1"),
        save_dir=str(SPLIT_DIR),
        rates=SPLIT_RATES,
        gap=SPLIT_GAP,
    )
    # 打包上傳快取（單一 tar；jpg 已壓縮，不需要 gzip）
    try:
        tar_path = Path("/content") / CACHE_FILE
        with tarfile.open(tar_path, "w") as tf:
            tf.add(SPLIT_DIR, arcname=SPLIT_DIR.name)
        size_gb = tar_path.stat().st_size / 1e9
        if size_gb <= CACHE_MAX_GB:
            api.create_repo(CACHE_REPO, repo_type="dataset", private=True, exist_ok=True)
            print(f"uploading split cache ({size_gb:.1f} GB) to {CACHE_REPO} ...")
            api.upload_file(
                path_or_fileobj=str(tar_path),
                path_in_repo=CACHE_FILE,
                repo_id=CACHE_REPO,
                repo_type="dataset",
            )
        else:
            print(f"split tar is {size_gb:.1f} GB > {CACHE_MAX_GB} GB, skipping cache upload")
        tar_path.unlink(missing_ok=True)
    except Exception as e:
        print("cache upload failed (non-fatal):", e)

# 產生訓練用 yaml（類別名稱直接取自官方 DOTAv1.yaml，避免順序寫錯）
from ultralytics.data.utils import check_det_dataset
import yaml

names = check_det_dataset("DOTAv1.yaml")["names"]
DATA_YAML.write_text(
    yaml.safe_dump(
        {"path": str(SPLIT_DIR), "train": "images/train", "val": "images/val", "names": names},
        sort_keys=False,
        allow_unicode=True,
    )
)
n_train = len(list((SPLIT_DIR / "images" / "train").glob("*")))
n_val = len(list((SPLIT_DIR / "images" / "val").glob("*")))
print(f"tiles: train={n_train}, val={n_val}; yaml -> {DATA_YAML}")

# %% [markdown]
# ## 3. HF checkpoint callback（與 repo 內 `src/obbkit/hf_checkpoint.py` 同步）

# %%
CKPT_DIR = "checkpoints"


def ensure_repo(repo_id: str, private: bool = True) -> str:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return repo_id


def install_hf_checkpoint_callback(model, repo_id: str, every_n_epochs: int = 2) -> None:
    def _upload(path, dest):
        p = Path(path)
        if p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=repo_id, repo_type="model")

    def on_model_save(trainer):
        if (trainer.epoch + 1) % every_n_epochs != 0:
            return
        try:
            _upload(trainer.last, f"{CKPT_DIR}/last.pt")
            _upload(Path(trainer.save_dir) / "results.csv", f"{CKPT_DIR}/results.csv")
            print(f"[hf_checkpoint] pushed last.pt @ epoch {trainer.epoch + 1} -> {repo_id}")
        except Exception as e:
            print(f"[hf_checkpoint] WARNING: push failed @ epoch {trainer.epoch + 1}: {e}")

    def on_train_end(trainer):
        try:
            _upload(trainer.best, f"{CKPT_DIR}/best.pt")
            _upload(trainer.last, f"{CKPT_DIR}/last.pt")
            _upload(Path(trainer.save_dir) / "results.csv", f"{CKPT_DIR}/results.csv")
            print(f"[hf_checkpoint] final push -> {repo_id}")
        except Exception as e:
            print(f"[hf_checkpoint] WARNING: final push failed: {e}")

    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_train_end", on_train_end)


def pull_resume_checkpoint(repo_id: str, dest_dir: str):
    remote = f"{CKPT_DIR}/last.pt"
    if not (api.repo_exists(repo_id) and api.file_exists(repo_id, remote)):
        return None
    return hf_hub_download(repo_id=repo_id, filename=remote, local_dir=dest_dir)

# %% [markdown]
# ## 4. 訓練（自動偵測 resume）

# %%
from ultralytics import YOLO

ensure_repo(MODEL_REPO, private=True)
ckpt = pull_resume_checkpoint(MODEL_REPO, "/content/resume")

if ckpt:
    print("found checkpoint on HF -> resuming previous run")
    model = YOLO(ckpt)
else:
    print("no checkpoint on HF -> fresh fine-tune from", MODEL)
    model = YOLO(MODEL)

install_hf_checkpoint_callback(model, MODEL_REPO, every_n_epochs=PUSH_EVERY)

if ckpt:
    model.train(resume=True)
else:
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        patience=PATIENCE,
        device=0,
        project="/content/runs",
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        **PRESETS[PRESET],
    )

# %% [markdown]
# ## 5. 評估：fine-tuned vs 官方 pretrained baseline（同一 val split、同條件）

# %%
BEST = f"/content/runs/{RUN_NAME}/weights/best.pt"

ft = YOLO(BEST).val(data=str(DATA_YAML), imgsz=IMGSZ, device=0, verbose=False)
base = YOLO(MODEL).val(data=str(DATA_YAML), imgsz=IMGSZ, device=0, verbose=False)

rows = [
    ("official " + MODEL + " (baseline)", base.box.map50, base.box.map),
    ("fine-tuned best.pt", ft.box.map50, ft.box.map),
]
print(f"\n{'model':<40}{'mAP50':>8}{'mAP50-95':>10}")
for name, m50, m in rows:
    print(f"{name:<40}{m50:>8.4f}{m:>10.4f}")
print(f"{'delta':<40}{ft.box.map50 - base.box.map50:>+8.4f}{ft.box.map - base.box.map:>+10.4f}")

# %% [markdown]
# ## 6. 上傳成果到 HF

# %%
run_dir = Path(f"/content/runs/{RUN_NAME}")
for f in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "results.png",
          "confusion_matrix_normalized.png"]:
    p = run_dir / f
    if p.is_file():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=Path(f).name, repo_id=MODEL_REPO, repo_type="model")
        print("uploaded", f)

# %%
import csv

with open(run_dir / "results.csv", newline="") as fh:
    hist = list(csv.DictReader(fh))

print("=" * 46)
print("=== PASTE BACK TO CLAUDE CODE ===")
print(f"gpu: {gpu} | preset: {PRESET} | imgsz: {IMGSZ}")
print(f"epochs completed: {len(hist)} / {EPOCHS} (patience={PATIENCE})")
print(f"train tiles: {n_train} | val tiles: {n_val} | rates: {SPLIT_RATES}")
print(f"baseline  ({MODEL}): mAP50={base.box.map50:.4f}  mAP50-95={base.box.map:.4f}")
print(f"fine-tuned (best.pt): mAP50={ft.box.map50:.4f}  mAP50-95={ft.box.map:.4f}")
print(f"HF model repo: {MODEL_REPO}")
print("=" * 46)

# %% [markdown]
# ## 完成後
# 1. 複製上方 `PASTE BACK` 區塊，貼回 Claude Code
# 2. **執行階段 → 中斷連線並刪除執行階段**（省運算單元）
