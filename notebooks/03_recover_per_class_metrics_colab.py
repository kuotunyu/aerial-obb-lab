# %% [markdown]
# # YOLO26m-OBB × DOTAv1 — 補回逐類別評估指標（Colab）
#
# 這份 notebook **只做 validation，不會重新訓練，也不會上傳或修改 Hugging Face 上的檔案**。
# 目的只有一個：用原本的 `best.pt`，依原版套件與切圖設定重建 DOTAv1 val split，補回先前
# 沒有保存的 `plane`、`ship`、`storage tank` 三類 fine-tuned mAP50 / mAP50-95。
#
# ## 使用方式
# 1. Colab「執行階段 → 變更執行階段類型」選 GPU；建議 A100。T4 通常也能做 validation，
#    若固定 batch=16 發生 OOM，請改用 A100，不要改 batch 後混用結果。
# 2. 不需要 HF token。請先將擁有者保存的原始 `best.pt` 上傳為 `/content/best.pt`；
#    這份 code-only release 不提供或自動下載 checkpoint。
# 3. 流程會從 Ultralytics 的固定 DOTAv1 release asset 下載原始資料，再用原參數
#    重切；請預留下載、切圖與完整驗證時間。
# 4. 只有在刻意重現既有證據時，才把 `ALLOW_HISTORICAL_GPU_RUN` 改成 `True` 後執行；這會下載
#    約 2 GB DOTA asset 並跑完整 validation。發布、檢查或展示本 repo 不需要執行。
#
# 完整性檢查不只看三個新數字：raw ZIP 與 checkpoint SHA-256、整體 mAP、原本已保存的
# 12 類、15 類順序、split 圖片數與 manifest 都必須吻合，結果才會標成 `PASS`。

# %% [markdown]
# ## 0. 固定原始評估設定

# %%
ALLOW_HISTORICAL_GPU_RUN = False

if not ALLOW_HISTORICAL_GPU_RUN:
    raise RuntimeError(
        "Historical full DOTAv1 validation is disabled by default. It downloads about 2 GB and "
        "requires an owner-supplied checkpoint plus GPU runtime. Review the release limitations, "
        "then set ALLOW_HISTORICAL_GPU_RUN=True only for an intentional evidence recovery run."
    )

HISTORICAL_MODEL_REVISION = "3f5705719a6e161fd105118fa8ba80b9a6cb1536"
WEIGHT_FILE = "best.pt"
EXPECTED_WEIGHT_SHA256 = "59727b5eccf16c07bde8535606da7f0b54c144266ed893cbb545ffe08789f188"

RAW_DATA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
RAW_RELEASE_API_URL = "https://api.github.com/repos/ultralytics/assets/releases/tags/v0.0.0"
RAW_ASSET_ID = 178_787_576
RAW_ASSET_SIZE = 1_988_510_024
RAW_ASSET_UPDATED_AT = "2024-07-10T14:47:04Z"
# 2026-07-15 從上述 immutable GitHub release asset 完整下載後計算。
EXPECTED_RAW_SHA256 = "59e84c52a8e7ee0ba89ee0679dc2a95833d6a11d0debba20ca01cbb11d58b816"
EXPECTED_RAW_TRAIN_IMAGES = 1_411
EXPECTED_RAW_VAL_IMAGES = 458

IMGSZ = 1024
SPLIT_RATES = [0.8, 1.2]
SPLIT_GAP = 500
EXPECTED_TRAIN_TILES = 62_030
EXPECTED_VAL_TILES = 21_271
EXPECTED_CLASS_COUNT = 15
AGGREGATE_TOLERANCE = 0.0015
CLASS_TOLERANCE = 0.0015  # 12 類舊紀錄為三位小數；另留少量跨 GPU 數值誤差

TARGET_CLASSES = ("plane", "ship", "storage tank")
# standalone validation 只保存了文件中的三位小數，故 gate 對這組 reported 值留明確 tolerance；
# 公開 results.csv 的 epoch 13 trainer-validation 精確值另外保留作參考，不混稱為 standalone 輸出。
EXPECTED_AGGREGATE = {"mAP50": 0.782, "mAP50-95": 0.631}
TRAINER_AGGREGATE_REFERENCE = {"mAP50": 0.78150, "mAP50-95": 0.63112}

# 原 Colab session 已保存的 12 類；重跑必須與這些值吻合，不能只看三個缺值。
EXPECTED_CAPTURED = {
    "helicopter": (0.833, 0.618),
    "soccer ball field": (0.660, 0.555),
    "small vehicle": (0.721, 0.571),
    "large vehicle": (0.824, 0.686),
    "ground track field": (0.750, 0.636),
    "bridge": (0.626, 0.407),
    "baseball diamond": (0.815, 0.604),
    "swimming pool": (0.706, 0.456),
    "tennis court": (0.918, 0.890),
    "roundabout": (0.668, 0.528),
    "harbor": (0.848, 0.591),
    "basketball court": (0.641, 0.587),
}

BASELINE_TARGETS = {
    "plane": (0.953, 0.863),
    "ship": (0.900, 0.754),
    "storage tank": (0.833, 0.705),
}

# 官方 DOTAv1 YAML 的類別順序；固定在 notebook 內，讓資料 YAML 不依賴另一個執行階段狀態。
DOTA_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool",
}

# %% [markdown]
# ## 1. 安裝與 GPU

# %%
# %pip install -q ultralytics==8.4.93

# %%
import torch
import ultralytics
import sys

assert ultralytics.__version__ == "8.4.93", (
    f"需要 ultralytics 8.4.93，實際為 {ultralytics.__version__}；請重新執行安裝格"
)
assert torch.cuda.is_available(), "沒有偵測到 GPU；請在 Colab 變更執行階段類型後重跑"

GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
# 原始 standalone val 未指定 batch，ultralytics 8.4.93 預設為 16；固定相同值。
VAL_BATCH = 16
print(
    f"GPU: {GPU} ({VRAM_GB:.0f} GB) | torch {torch.__version__} | "
    f"CUDA {torch.version.cuda} | ultralytics {ultralytics.__version__} | val batch {VAL_BATCH}"
)
print("Python:", sys.version.replace("\n", " "))

# %% [markdown]
# ## 2. 取得並驗證原始 `best.pt`
#
# 必須由擁有者先上傳 `/content/best.pt`。流程不會從遠端獲取權重。
# SHA-256 必須與原始 epoch 13 checkpoint 完全一致。

# %%
import hashlib
import shutil
from pathlib import Path

WEIGHTS = Path("/content/best.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if not WEIGHTS.is_file():
    raise RuntimeError(
        "請先上傳已校驗的原始 best.pt 到 /content/best.pt，再從本格重跑。"
    )

actual_weight_sha256 = sha256_file(WEIGHTS)
assert actual_weight_sha256 == EXPECTED_WEIGHT_SHA256, (
    "best.pt SHA-256 不符，不能用來回填指標。"
    f"\nexpected: {EXPECTED_WEIGHT_SHA256}\nactual:   {actual_weight_sha256}"
)
print(f"weights verified: {WEIGHTS} ({WEIGHTS.stat().st_size / 1e6:.1f} MB)")
print("sha256:", actual_weight_sha256)

# %% [markdown]
# ## 3. 從固定 DOTAv1 release 重建相同設定的 split
#
# 原訓練用的 private tar 沒有版本化在本 repo，不能把一個可變、無已知雜湊的 cache 當成
# 「完全相同」的證據。本流程改從 `ultralytics==8.4.93` 內固定的 DOTAv1 release URL 重建，
# 並驗證 raw ZIP SHA-256、raw/split 圖片數、15 類標註存在性，以及 val 檔名、圖片大小與
# label 內容 manifest。
# manifest 會寫入 provenance marker；同一個 Colab runtime 斷線重跑時只有 manifest 仍吻合才沿用。

# %%
import json
import stat
import zipfile
from datetime import datetime, timezone

import yaml
from ultralytics import settings

DATASETS = Path("/content/datasets")
RAW_DIR = DATASETS / "DOTAv1"
RAW_ARCHIVE = Path("/content/DOTAv1.zip")
SPLIT_DIR = DATASETS / "DOTAv1-split"
DATA_YAML = DATASETS / "DOTAv1-split.yaml"
PROVENANCE_FILE = SPLIT_DIR / ".recovery_split_provenance.json"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

settings.update({"datasets_dir": str(DATASETS)})


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in folder.iterdir())


def split_counts() -> dict[str, int]:
    return {
        "train": count_images(SPLIT_DIR / "images" / "train"),
        "val": count_images(SPLIT_DIR / "images" / "val"),
    }


def split_structure_ready() -> bool:
    required = (
        SPLIT_DIR / "images" / "train",
        SPLIT_DIR / "images" / "val",
        SPLIT_DIR / "labels" / "train",
        SPLIT_DIR / "labels" / "val",
    )
    return all(folder.is_dir() for folder in required)


def split_ready() -> bool:
    return split_structure_ready() and split_counts() == {
        "train": EXPECTED_TRAIN_TILES,
        "val": EXPECTED_VAL_TILES,
    }


def remove_managed_directory(path: Path) -> None:
    if not path.exists():
        return
    # 只允許刪除這份 notebook 自己管理的 /content/datasets 子目錄。
    if DATASETS.resolve() not in path.resolve().parents:
        raise RuntimeError(f"拒絕移除非預期路徑：{path}")
    shutil.rmtree(path)


def remove_incomplete_split() -> None:
    remove_managed_directory(SPLIT_DIR)


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"DOTAv1.zip 含不安全路徑：{member.filename}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"DOTAv1.zip 不應包含 symbolic link：{member.filename}")
        archive.extractall(destination)


def val_manifest_sha256() -> str:
    """Hash val filenames, image sizes, and exact label bytes without rereading all image pixels."""
    digest = hashlib.sha256(b"dotav1-recovery-val-manifest-v1\n")
    image_dir = SPLIT_DIR / "images" / "val"
    label_dir = SPLIT_DIR / "labels" / "val"
    images = sorted(
        (path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    for image_path in images:
        digest.update(f"image\0{image_path.name}\0{image_path.stat().st_size}\0".encode())
    for label_path in sorted(label_dir.glob("*.txt"), key=lambda path: path.name):
        digest.update(f"label\0{label_path.name}\0".encode())
        digest.update(label_path.read_bytes() + b"\0")
    return digest.hexdigest()


def val_label_class_counts() -> dict[str, int]:
    counts_by_id = {class_id: 0 for class_id in DOTA_NAMES}
    for label_path in sorted((SPLIT_DIR / "labels" / "val").glob("*.txt")):
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                class_id = int(float(line.split()[0]))
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"無法解析 {label_path.name}:{line_number}") from exc
            if class_id not in counts_by_id:
                raise RuntimeError(f"{label_path.name}:{line_number} 含未知 class_id={class_id}")
            counts_by_id[class_id] += 1
    assert all(count > 0 for count in counts_by_id.values()), "val split 並未包含完整 15 類標註"
    return {DOTA_NAMES[class_id]: counts_by_id[class_id] for class_id in DOTA_NAMES}


def expected_provenance_config() -> dict:
    return {
        "schema_version": 1,
        "ultralytics": "8.4.93",
        "raw_data_url": RAW_DATA_URL,
        "raw_release_api_url": RAW_RELEASE_API_URL,
        "raw_asset_id": RAW_ASSET_ID,
        "raw_asset_size": RAW_ASSET_SIZE,
        "raw_asset_updated_at": RAW_ASSET_UPDATED_AT,
        "raw_archive_sha256": EXPECTED_RAW_SHA256,
        "imgsz": IMGSZ,
        "split_rates": SPLIT_RATES,
        "split_gap": SPLIT_GAP,
        "train_tiles": EXPECTED_TRAIN_TILES,
        "val_tiles": EXPECTED_VAL_TILES,
    }


def reusable_split_manifest() -> str | None:
    if not split_ready() or not PROVENANCE_FILE.is_file():
        return None
    try:
        provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
        if any(provenance.get(key) != value for key, value in expected_provenance_config().items()):
            return None
        actual_manifest = val_manifest_sha256()
        if actual_manifest != provenance.get("val_manifest_sha256"):
            return None
        if val_label_class_counts() != provenance.get("val_label_class_counts"):
            return None
        return actual_manifest
    except (OSError, ValueError, RuntimeError, AssertionError):
        return None


VAL_MANIFEST_SHA256 = reusable_split_manifest()
if VAL_MANIFEST_SHA256:
    RAW_ARCHIVE_SHA256 = EXPECTED_RAW_SHA256
    DATASET_SOURCE = "verified_sha256_source_rebuild"
    print("reusing notebook-generated split:", split_counts())
else:
    if SPLIT_DIR.exists():
        print("removing unverified/incomplete split:", split_counts())
        remove_incomplete_split()

    from ultralytics.data.split_dota import split_trainval
    from ultralytics.utils.checks import check_yaml
    from urllib.request import Request, urlopen

    print("downloading the fixed DOTAv1 release and rebuilding the original split settings ...")
    request = Request(RAW_RELEASE_API_URL, headers={"User-Agent": "yolo26-dota-obb-recovery"})
    with urlopen(request, timeout=30) as response:
        release_metadata = json.load(response)
    matching_assets = [asset for asset in release_metadata["assets"] if asset["name"] == "DOTAv1.zip"]
    assert len(matching_assets) == 1, "GitHub release 未找到唯一的 DOTAv1.zip asset"
    raw_asset = matching_assets[0]
    assert raw_asset["id"] == RAW_ASSET_ID, "DOTAv1 release asset ID 已改變"
    assert raw_asset["size"] == RAW_ASSET_SIZE, "DOTAv1 release asset size 已改變"
    assert raw_asset["updated_at"] == RAW_ASSET_UPDATED_AT, "DOTAv1 release asset timestamp 已改變"
    assert raw_asset["browser_download_url"] == RAW_DATA_URL, "DOTAv1 release asset URL 已改變"

    official_definition = yaml.safe_load(
        Path(check_yaml("DOTAv1.yaml")).read_text(encoding="utf-8")
    )
    official_names = {int(class_id): name for class_id, name in official_definition["names"].items()}
    assert official_definition.get("download") == RAW_DATA_URL, "8.4.93 DOTAv1 download URL 不符"
    assert official_names == DOTA_NAMES, "8.4.93 DOTAv1 類別名稱或順序不符"

    # 不沿用 /content 裡來源不明、但張數剛好相同的 raw dataset。每次建立新的 split 都從
    # immutable GitHub asset 重新下載，並在解壓前驗完整 archive SHA-256。
    remove_managed_directory(RAW_DIR)
    RAW_ARCHIVE.unlink(missing_ok=True)
    try:
        torch.hub.download_url_to_file(
            RAW_DATA_URL,
            str(RAW_ARCHIVE),
            hash_prefix=EXPECTED_RAW_SHA256,
            progress=True,
        )
        assert RAW_ARCHIVE.stat().st_size == RAW_ASSET_SIZE, "DOTAv1.zip 實際檔案大小不符"
        RAW_ARCHIVE_SHA256 = sha256_file(RAW_ARCHIVE)
        assert RAW_ARCHIVE_SHA256 == EXPECTED_RAW_SHA256, "DOTAv1.zip SHA-256 不符"
        safe_extract_zip(RAW_ARCHIVE, DATASETS)
    finally:
        RAW_ARCHIVE.unlink(missing_ok=True)  # 解壓後立刻釋放約 2 GB 空間

    raw_root = RAW_DIR
    assert raw_root.is_dir(), f"DOTAv1.zip 未產生預期目錄：{raw_root}"
    raw_counts = {
        "train": count_images(raw_root / "images" / "train"),
        "val": count_images(raw_root / "images" / "val"),
    }
    assert raw_counts == {
        "train": EXPECTED_RAW_TRAIN_IMAGES,
        "val": EXPECTED_RAW_VAL_IMAGES,
    }, f"raw DOTAv1 圖片數不符：{raw_counts}"

    split_trainval(
        data_root=str(raw_root),
        save_dir=str(SPLIT_DIR),
        rates=SPLIT_RATES,
        gap=SPLIT_GAP,
    )
    assert split_ready(), f"重建後 split 結構或圖片數不符：{split_counts()}"

    VAL_MANIFEST_SHA256 = val_manifest_sha256()
    VAL_LABEL_CLASS_COUNTS = val_label_class_counts()
    provenance = {
        **expected_provenance_config(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "val_manifest_sha256": VAL_MANIFEST_SHA256,
        "val_label_class_counts": VAL_LABEL_CLASS_COUNTS,
    }
    PROVENANCE_FILE.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    DATASET_SOURCE = "rebuilt_from_sha256_verified_archive"

counts = split_counts()
assert split_structure_ready(), "split 缺少 images/labels 的 train/val 目錄，停止評估"
assert counts == {"train": EXPECTED_TRAIN_TILES, "val": EXPECTED_VAL_TILES}, (
    "split 圖片數與原始訓練不符，停止評估。"
    f" expected train/val={EXPECTED_TRAIN_TILES}/{EXPECTED_VAL_TILES}, actual={counts}"
)
VAL_LABEL_CLASS_COUNTS = val_label_class_counts()
assert VAL_MANIFEST_SHA256 == val_manifest_sha256(), "val manifest 在資料準備期間發生變化"

DATASETS.mkdir(parents=True, exist_ok=True)
DATA_YAML.write_text(
    yaml.safe_dump(
        {
            "path": str(SPLIT_DIR),
            "train": "images/train",
            "val": "images/val",
            "names": DOTA_NAMES,
        },
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)
print(f"split verified: train={counts['train']}, val={counts['val']} | source={DATASET_SOURCE}")
print("val manifest sha256:", VAL_MANIFEST_SHA256)
print("dataset yaml:", DATA_YAML)

# %% [markdown]
# ## 4. 只跑一次完整 15 類 validation
#
# 不能用 `classes=` 只篩三類：per-class AP 必須放在與原評估相同的完整資料與類別情境中計算。
# 這裡保留原始 `val(data=..., imgsz=1024, device=0, verbose=False)` 條件，並將當時省略時的
# `ultralytics==8.4.93` 預設 batch=16 明確寫出；不開 FP16、TTA 或類別篩選。

# %%
from ultralytics import YOLO

torch.cuda.empty_cache()
model = YOLO(str(WEIGHTS))
metrics = model.val(
    data=str(DATA_YAML),
    imgsz=IMGSZ,
    device=0,
    batch=VAL_BATCH,
    plots=False,
    verbose=False,
)

print(
    f"validation complete: mAP50={float(metrics.box.map50):.6f}, "
    f"mAP50-95={float(metrics.box.map):.6f}"
)

# %% [markdown]
# ## 5. 匯出 15 類結果並執行完整性閘門
#
# Ultralytics 會在建立 validation dataset 時驗證標註並移除重複列，因此磁碟上的原始
# `.txt` 列數只應是有效 instances 的上限。精確的數量 gate 改以同一次 validation
# 產生的 `labels/val.cache`（loader 驗證後資料）對照 `metrics.nt_per_class` 與
# `metrics.nt_per_image`；raw 列數仍保留作 provenance 與診斷。

# %%
import csv

import numpy as np


def metric_class_name(names, class_id: int) -> str:
    return str(names[class_id])


class_ids = np.asarray(metrics.box.ap_class_index, dtype=int).reshape(-1)
precision_values = np.asarray(metrics.box.p, dtype=float).reshape(-1)
recall_values = np.asarray(metrics.box.r, dtype=float).reshape(-1)
f1_values = np.asarray(metrics.box.f1, dtype=float).reshape(-1)
ap50_values = np.asarray(metrics.box.ap50, dtype=float).reshape(-1)
ap_values = np.asarray(metrics.box.ap, dtype=float).reshape(-1)
images_per_class = np.asarray(metrics.nt_per_image, dtype=int).reshape(-1)
instances_per_class = np.asarray(metrics.nt_per_class, dtype=int).reshape(-1)


def validator_cache_class_counts(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Count post-verification instances and class-bearing images from Ultralytics' val cache."""
    if not cache_path.is_file():
        raise RuntimeError(f"找不到 validation cache：{cache_path}")
    try:
        cache = np.load(cache_path, allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f"無法讀取 validation cache：{cache_path}") from exc

    labels = cache.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError("validation cache 沒有可核對的 labels")

    instance_counts = np.zeros(EXPECTED_CLASS_COUNT, dtype=int)
    image_counts = np.zeros(EXPECTED_CLASS_COUNT, dtype=int)
    for record in labels:
        classes = np.asarray(record.get("cls", []), dtype=float).reshape(-1)
        if not len(classes):
            continue
        class_ids_for_image = classes.astype(int)
        if not np.array_equal(classes, class_ids_for_image):
            raise RuntimeError("validation cache 含非整數 class id")
        if np.any((class_ids_for_image < 0) | (class_ids_for_image >= EXPECTED_CLASS_COUNT)):
            raise RuntimeError("validation cache 含超出範圍的 class id")
        instance_counts += np.bincount(class_ids_for_image, minlength=EXPECTED_CLASS_COUNT)
        image_counts += np.bincount(
            np.unique(class_ids_for_image), minlength=EXPECTED_CLASS_COUNT
        )
    return instance_counts, image_counts


VAL_CACHE_PATH = SPLIT_DIR / "labels" / "val.cache"
cache_instances_per_class, cache_images_per_class = validator_cache_class_counts(VAL_CACHE_PATH)

assert len(class_ids) == len(precision_values) == len(recall_values) == len(f1_values), (
    "Ultralytics per-class precision/recall/F1 長度不一致"
)
assert len(class_ids) == len(ap50_values) == len(ap_values), "Ultralytics per-class AP 長度不一致"
assert len(images_per_class) == len(instances_per_class) == EXPECTED_CLASS_COUNT, (
    "Ultralytics class count vectors 長度不符"
)

rows = sorted(
    [
        {
            "class_id": int(class_id),
            "class": metric_class_name(metrics.names, int(class_id)),
            "images": int(images_per_class[int(class_id)]),
            "instances": int(instances_per_class[int(class_id)]),
            "precision": float(precision_values[index]),
            "recall": float(recall_values[index]),
            "f1": float(f1_values[index]),
            "mAP50": float(ap50_values[index]),
            "mAP50-95": float(ap_values[index]),
        }
        for index, class_id in enumerate(class_ids)
    ],
    key=lambda row: row["class_id"],
)
for row in rows:
    raw_label_lines = int(VAL_LABEL_CLASS_COUNTS[row["class"]])
    row["raw_label_lines"] = raw_label_lines
    row["deduplicated_or_filtered_lines"] = raw_label_lines - row["instances"]
by_class = {row["class"]: row for row in rows}

class_order_ok = (
    len(rows) == EXPECTED_CLASS_COUNT
    and [row["class_id"] for row in rows] == list(range(EXPECTED_CLASS_COUNT))
    and [row["class"] for row in rows] == list(DOTA_NAMES.values())
)

aggregate = {
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
}
aggregate_diffs = {
    key: abs(aggregate[key] - expected) for key, expected in EXPECTED_AGGREGATE.items()
}
aggregate_ok = all(diff <= AGGREGATE_TOLERANCE for diff in aggregate_diffs.values())

captured_checks = []
for class_name, (expected_ap50, expected_ap) in EXPECTED_CAPTURED.items():
    actual = by_class.get(class_name)
    captured_checks.append(
        {
            "class": class_name,
            "mAP50_abs_diff": None if actual is None else abs(actual["mAP50"] - expected_ap50),
            "mAP50-95_abs_diff": None if actual is None else abs(actual["mAP50-95"] - expected_ap),
        }
    )

captured_ok = all(
    check["mAP50_abs_diff"] is not None
    and check["mAP50_abs_diff"] <= CLASS_TOLERANCE
    and check["mAP50-95_abs_diff"] <= CLASS_TOLERANCE
    for check in captured_checks
)
targets_present = all(class_name in by_class for class_name in TARGET_CLASSES)
weight_sha256_ok = actual_weight_sha256 == EXPECTED_WEIGHT_SHA256
raw_archive_sha256_ok = RAW_ARCHIVE_SHA256 == EXPECTED_RAW_SHA256
split_counts_ok = counts == {"train": EXPECTED_TRAIN_TILES, "val": EXPECTED_VAL_TILES}
split_provenance_ok = reusable_split_manifest() == VAL_MANIFEST_SHA256
metric_instances_match_cache = np.array_equal(instances_per_class, cache_instances_per_class)
metric_images_match_cache = np.array_equal(images_per_class, cache_images_per_class)
# `verify_image_label()` in ultralytics 8.4.93 removes duplicate label rows before validation;
# raw text lines are therefore an upper bound, not a value that should equal `nt_per_class`.
raw_line_upper_bound_ok = all(
    0 < row["instances"] <= row["raw_label_lines"] for row in rows
)
instances_exactly_equal_raw_lines = all(
    row["instances"] == row["raw_label_lines"] for row in rows
)
integrity_pass = (
    weight_sha256_ok
    and raw_archive_sha256_ok
    and split_counts_ok
    and split_structure_ready()
    and split_provenance_ok
    and class_order_ok
    and metric_instances_match_cache
    and metric_images_match_cache
    and raw_line_upper_bound_ok
    and aggregate_ok
    and captured_ok
    and targets_present
)

integrity = {
    "status": "PASS" if integrity_pass else "FAIL",
    "aggregate_tolerance": AGGREGATE_TOLERANCE,
    "per_class_tolerance": CLASS_TOLERANCE,
    "weight_sha256_ok": weight_sha256_ok,
    "raw_archive_sha256_ok": raw_archive_sha256_ok,
    "split_structure_ok": split_structure_ready(),
    "split_counts_ok": split_counts_ok,
    "split_provenance_ok": split_provenance_ok,
    "class_order_ok": class_order_ok,
    "metric_instances_match_validator_cache": metric_instances_match_cache,
    "metric_images_match_validator_cache": metric_images_match_cache,
    "metric_instances_within_raw_label_lines": raw_line_upper_bound_ok,
    "raw_label_lines_equal_validator_instances": instances_exactly_equal_raw_lines,
    "instance_accounting": [
        {
            "class": row["class"],
            "raw_label_lines": row["raw_label_lines"],
            "validator_instances": row["instances"],
            "validator_cache_instances": int(cache_instances_per_class[row["class_id"]]),
            "validator_images": row["images"],
            "validator_cache_images": int(cache_images_per_class[row["class_id"]]),
            "deduplicated_or_filtered_lines": row["deduplicated_or_filtered_lines"],
        }
        for row in rows
    ],
    "aggregate_ok": aggregate_ok,
    "aggregate_abs_diff": aggregate_diffs,
    "captured_12_classes_ok": captured_ok,
    "captured_12_class_abs_diff": captured_checks,
    "targets_present": targets_present,
}

OUTPUT_DIR = Path("/content/per_class_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "per_class_metrics.csv"
JSON_PATH = OUTPUT_DIR / "per_class_metrics.json"
MARKDOWN_PATH = OUTPUT_DIR / "missing_class_rows.md"

with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=(
            "class_id",
            "class",
            "images",
            "instances",
            "raw_label_lines",
            "deduplicated_or_filtered_lines",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50-95",
        ),
    )
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_distribution": "not included; owner-supplied checkpoint required",
    "historical_model_revision": HISTORICAL_MODEL_REVISION,
    "weights": {
        "file": WEIGHT_FILE,
        "sha256": actual_weight_sha256,
    },
    "environment": {
        "gpu": GPU,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    },
    "dataset": {
        "name": "DOTAv1",
        "source": DATASET_SOURCE,
        "raw_data_url": RAW_DATA_URL,
        "raw_release_api_url": RAW_RELEASE_API_URL,
        "raw_asset_id": RAW_ASSET_ID,
        "raw_asset_size": RAW_ASSET_SIZE,
        "raw_asset_updated_at": RAW_ASSET_UPDATED_AT,
        "raw_archive_sha256": RAW_ARCHIVE_SHA256,
        "imgsz": IMGSZ,
        "split_rates": SPLIT_RATES,
        "split_gap": SPLIT_GAP,
        "train_tiles": counts["train"],
        "val_tiles": counts["val"],
        "val_manifest_sha256": VAL_MANIFEST_SHA256,
        "val_label_class_counts": VAL_LABEL_CLASS_COUNTS,
        "val_effective_instance_counts": {
            DOTA_NAMES[class_id]: int(cache_instances_per_class[class_id])
            for class_id in DOTA_NAMES
        },
        "val_effective_image_counts": {
            DOTA_NAMES[class_id]: int(cache_images_per_class[class_id])
            for class_id in DOTA_NAMES
        },
    },
    "aggregate": aggregate,
    "aggregate_reference": {
        "standalone_reported_3dp": EXPECTED_AGGREGATE,
        "public_results_csv_epoch13_trainer_validation": TRAINER_AGGREGATE_REFERENCE,
    },
    "integrity": integrity,
    "per_class": rows,
}
JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def signed_delta(value: float) -> str:
    return f"{value:+.3f}"


markdown_rows = []
for class_name in TARGET_CLASSES:
    baseline_ap50, baseline_ap = BASELINE_TARGETS[class_name]
    row = by_class.get(class_name)
    if row is None:
        markdown_rows.append(
            f"| {class_name} | {baseline_ap50:.3f} | — | — | {baseline_ap:.3f} | — | — |"
        )
    else:
        markdown_rows.append(
            f"| {class_name} | {baseline_ap50:.3f} | {row['mAP50']:.3f} | "
            f"{signed_delta(row['mAP50'] - baseline_ap50)} | {baseline_ap:.3f} | "
            f"{row['mAP50-95']:.3f} | {signed_delta(row['mAP50-95'] - baseline_ap)} |"
        )

MARKDOWN_PATH.write_text(
    "# Candidate rows for docs/training_results.md\n\n"
    f"Integrity gate: **{integrity['status']}**\n\n"
    "| class | baseline mAP50 | fine-tuned mAP50 | Δ mAP50 | baseline mAP50-95 | "
    "fine-tuned mAP50-95 | Δ mAP50-95 |\n"
    "|---|---:|---:|---:|---:|---:|---:|\n"
    + "\n".join(markdown_rows)
    + "\n",
    encoding="utf-8",
)

BUNDLE_BASE = Path("/content/per_class_metrics_bundle")
BUNDLE_PATH = Path(shutil.make_archive(str(BUNDLE_BASE), "zip", root_dir=OUTPUT_DIR))

print("\n" + "=" * 64)
print("=== PASTE BACK TO CODEX ===")
print(
    f"integrity_gate: {integrity['status']} "
    f"(aggregate_tol={AGGREGATE_TOLERANCE}, per_class_tol={CLASS_TOLERANCE})"
)
print(f"checkpoint_sha256: {actual_weight_sha256}")
print(f"raw_archive_sha256: {RAW_ARCHIVE_SHA256}")
print(f"split: train={counts['train']} val={counts['val']} rates={SPLIT_RATES} gap={SPLIT_GAP}")
print(f"val_manifest_sha256: {VAL_MANIFEST_SHA256}")
print(f"aggregate: mAP50={aggregate['mAP50']:.6f} mAP50-95={aggregate['mAP50-95']:.6f}")
for class_name in TARGET_CLASSES:
    row = by_class.get(class_name)
    if row is None:
        print(f"{class_name}: MISSING")
    else:
        print(f"{class_name}: mAP50={row['mAP50']:.6f} mAP50-95={row['mAP50-95']:.6f}")
print(f"artifacts: {BUNDLE_PATH}")
print("=" * 64)

if integrity_pass:
    print("PASS：這三類數字可用來回填 docs/training_results.md。")
else:
    print("FAIL：先保留輸出供診斷，不要把這三類數字寫入正式結果。")
    print(json.dumps(integrity, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 6. 下載結果
#
# zip 內含完整 15 類 CSV、帶環境與完整性檢查資訊的 JSON，以及可直接回填文件的三列表格。
# 若瀏覽器擋住自動下載，也可從 Colab 左側檔案面板下載
# `/content/per_class_metrics_bundle.zip`。

# %%
from google.colab import files

files.download(str(BUNDLE_PATH))
