# %% [markdown]
# # YOLO26m-OBB 部署 benchmark — PyTorch vs ONNX Runtime vs TensorRT FP16
#
# 從 HF 下載 fine-tune 好的 `best.pt` → 匯出 **ONNX** 與 **TensorRT FP16 engine** →
# 驗證匯出精度沒有掉（防 YOLO26 e2e 匯出已知 issue）→ 量測三種後端的 latency / FPS。
#
# ## 使用步驟
# 1. 執行階段 → **T4 GPU**（建議：Turing 世代、省運算單元、官方 speed 基準也用 T4；A100 亦可）
# 2. 左側 🔑 Secrets：`HF_TOKEN`（與訓練 notebook 相同）
# 3. 全部執行（TensorRT 安裝 + engine build 約 10–20 分鐘）
# 4. 把最後的 `=== PASTE BACK ===` 區塊貼回 Claude Code
# 5. 跑完記得中斷連線並刪除執行階段
#
# > 注意:TensorRT engine 綁定 build 時的 GPU 型號與 TensorRT 版本，benchmark 數字都會標注環境。

# %% [markdown]
# ## 0. 設定

# %%
HF_MODEL_REPO_NAME = "yolo26m-obb-dota"  # -> {你的帳號}/yolo26m-obb-dota
WEIGHT_FILE = "best.pt"                  # 訓練 notebook 上傳的檔名
IMGSZ = 1024
N_WARMUP = 20
N_RUNS = 100
PARITY_DATA = "dota8.yaml"               # 匯出精度 parity 檢查用（小而快；抓大趨勢）
PARITY_TOL = 0.01                        # exported vs PyTorch 的 mAP50 容許差距（1 個百分點）

# %% [markdown]
# ## 1. 安裝與登入

# %%
# %pip install -q ultralytics==8.4.93

# %%
from google.colab import userdata
from huggingface_hub import HfApi, hf_hub_download, login

login(userdata.get("HF_TOKEN"))
api = HfApi()
HF_USER = api.whoami()["name"]
MODEL_REPO = f"{HF_USER}/{HF_MODEL_REPO_NAME}"

import torch

assert torch.cuda.is_available(), "沒有 GPU！請確認執行階段類型"
GPU = torch.cuda.get_device_name(0)
print("GPU:", GPU, "| repo:", MODEL_REPO)

# %% [markdown]
# ## 2. 下載權重、匯出 ONNX 與 TensorRT FP16

# %%
from pathlib import Path

from ultralytics import YOLO

best = hf_hub_download(MODEL_REPO, WEIGHT_FILE, local_dir="/content/weights")
print("weights:", best)

onnx_path = YOLO(best).export(format="onnx", imgsz=IMGSZ, device=0)
print("onnx:", onnx_path)

# TensorRT：ultralytics 會自動 pip 安裝 tensorrt 並 build engine（需要幾分鐘）
engine_path = YOLO(best).export(format="engine", imgsz=IMGSZ, half=True, device=0)
print("engine:", engine_path)

# %% [markdown]
# ## 3. 匯出精度 parity 檢查
#
# YOLO26 end-to-end 匯出有已知 issue（ultralytics#23397 等），先確認 exported 模型
# 的 mAP 沒有崩掉再看速度。這裡用 dota8 val 快速比對（同工具同條件的相對比較）。

# %%
def val_map50(weights):
    m = YOLO(weights, task="obb")  # exported formats need the explicit task hint
    r = m.val(data=PARITY_DATA, imgsz=IMGSZ, device=0, verbose=False)
    return float(r.box.map50), float(r.box.map)

pt_map50, pt_map = val_map50(best)
onnx_map50, onnx_map = val_map50(onnx_path)
trt_map50, trt_map = val_map50(engine_path)

print(f"{'backend':<12}{'mAP50':>8}{'mAP50-95':>10}")
for name, m50, m in [("PyTorch", pt_map50, pt_map), ("ONNX", onnx_map50, onnx_map), ("TensorRT", trt_map50, trt_map)]:
    print(f"{name:<12}{m50:>8.4f}{m:>10.4f}")

parity_ok = abs(onnx_map50 - pt_map50) < PARITY_TOL and abs(trt_map50 - pt_map50) < PARITY_TOL
print("PARITY:", "OK" if parity_ok else "⚠️ FAILED — exported model accuracy diverged, do not trust speed table")

# %% [markdown]
# ## 4. Latency / FPS benchmark（batch=1）

# %%
import time

import numpy as np

# 固定一張 1024x1024 輸入，避免 IO / letterbox 差異干擾
img = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

def bench(weights, label):
    m = YOLO(weights, task="obb")
    for _ in range(N_WARMUP):
        m.predict(img, imgsz=IMGSZ, device=0, verbose=False)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        m.predict(img, imgsz=IMGSZ, device=0, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "backend": label,
        "size_mb": round(Path(weights).stat().st_size / 1e6, 1),
        "mean_ms": round(float(times.mean()), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(1000 / float(times.mean()), 1),
    }

rows = [
    bench(best, "PyTorch (FP32)"),
    bench(onnx_path, "ONNX Runtime GPU"),
    bench(engine_path, "TensorRT FP16"),
]

import polars as pl  # ultralytics 相依已含 polars

print(pl.DataFrame(rows))

# %% [markdown]
# ## 5. 上傳 ONNX 到 HF、輸出結果區塊

# %%
api.upload_file(path_or_fileobj=onnx_path, path_in_repo=Path(onnx_path).name,
                repo_id=MODEL_REPO, repo_type="model")
print("uploaded", Path(onnx_path).name, "->", MODEL_REPO)

# %%
import tensorrt, onnxruntime

print("=" * 50)
print("=== PASTE BACK TO CLAUDE CODE ===")
print(f"gpu: {GPU} | imgsz: {IMGSZ} | torch {torch.__version__} | ort {onnxruntime.__version__} | trt {tensorrt.__version__}")
print(f"parity({PARITY_DATA}): PT mAP50={pt_map50:.4f} / ONNX {onnx_map50:.4f} / TRT {trt_map50:.4f} -> {'OK' if parity_ok else 'FAILED'}")
print(f"{'backend':<18}{'size(MB)':>9}{'mean ms':>9}{'p50 ms':>8}{'p95 ms':>8}{'FPS':>7}")
for r in rows:
    print(f"{r['backend']:<18}{r['size_mb']:>9}{r['mean_ms']:>9}{r['p50_ms']:>8}{r['p95_ms']:>8}{r['fps']:>7}")
print("=" * 50)

# %% [markdown]
# ## 完成後
# 1. 複製 `PASTE BACK` 區塊貼回 Claude Code
# 2. 執行階段 → 中斷連線並刪除執行階段
