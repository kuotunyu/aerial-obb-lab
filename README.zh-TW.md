# YOLO26 OBB × DOTA：航拍旋轉框偵測，從訓練到部署

> 🚧 進行中 — 完整計畫見 [docs/PLAN.md](docs/PLAN.md)。

以 **YOLO26-OBB**（旋轉框）在 **DOTAv1** 航拍資料集上 fine-tune，展示完整生命週期：
Colab A100 訓練 → 與官方 baseline 對照評估 → 用數據回答「為什麼航拍場景需要 OBB」→ ONNX / TensorRT FP16 匯出與三框架 latency benchmark → Gradio demo + Hugging Face Space。

## 進度

| Phase | 內容 | 位置 | 狀態 |
|---|---|---|---|
| 0 | 環境與骨架 | 本機 | ✅ |
| 1 | DOTA8 smoke test | 本機（RTX 2070） | ✅ |
| 2 | DOTAv1 正式 fine-tune | Colab A100 | ✅ |
| 3 | 對照官方 baseline 評估 | Colab + 本機 | ✅ |
| 4 | 「為什麼需要 OBB」量化分析 | 本機 | ✅ |
| 5 | ONNX / TensorRT 匯出與 benchmark | Colab GPU | ✅ |
| 6 | Gradio demo + HF Space（CPU） | 本機 + HF | ✅ |
| 7 | 文件、model card、收尾 | — | ⬜ |

## 評估：Fine-tuned vs 官方 Baseline

在 Colab A100 上用重新切過的 DOTAv1（`split_dota` 尺度 `[0.8, 1.2]`）fine-tune
`yolo26m-obb.pt`，跑了 28/30 個 epoch、由 `patience=15` 觸發提早停止。完整逐類別對照、
訓練曲線分析與混淆矩陣發現見 [docs/training_results.md](docs/training_results.md)。

| 模型 | split | mAP50 | mAP50-95 |
|---|---|---:|---:|
| yolo26m-obb.pt（官方公布數字） | DOTAv1 test | 81.0 | 55.3 |
| yolo26m-obb.pt（官方權重，本專案實測） | DOTAv1 val（本專案切法） | 78.2 | 63.3 |
| fine-tuned `best.pt` | DOTAv1 val（本專案切法） | 78.2 | 63.1 |

**只有下面兩行可以直接比較**——同一個 val split、同樣的切圖方式、同一套驗證流程、同條件。
第一行是官方在 test split 上、用 DOTA 官方評測工具算出的數字，放在這裡僅供參考，不是拿來
比較用的（test/val 這個落差在 mAP50 跟 mAP50-95 上方向不一致，原因見
[docs/training_results.md](docs/training_results.md)）。

在同條件下比較，fine-tune 幾乎沒有帶來進步（Δ mAP50 -0.05 個百分點、Δ mAP50-95 -0.13 個
百分點）。這是預期中的結果，不是訓練失敗：`yolo26m-obb.pt` 本來就是官方在 DOTAv1 上訓練過
的權重，這次等於是「繼續訓練一個已經收斂的模型」而不是「帶模型認識新領域」，這個結果量化
的正是這種情境下的邊際效益上限，不是要展示一次大幅進步。

## 為什麼需要旋轉框？（用數據說話）

在 **DOTAv1 驗證集 456 張圖、28,853 個標註物件**上實測（完整表格見
[docs/analysis_results.md](docs/analysis_results.md)，重現腳本
[scripts/obb_analysis.py](scripts/obb_analysis.py)）：

**1. 水平框吞掉大量背景。** 「軸對齊外接框面積 ÷ 旋轉框面積」：

| 類別 | 平均 | p90 | | 對照組（圓形） | 平均 |
|---|---:|---:|---|---|---:|
| bridge | **2.43×** | 3.71× | | roundabout | 1.02× |
| harbor | **2.15×** | 3.66× | | storage tank | 1.00× |
| large vehicle | **2.14×** | 3.21× | | | |
| ship | **1.95×** | 2.69× | | | |

細長且任意旋轉的目標會讓水平框膨脹約 2 倍；而圓形類別（儲油槽、圓環）維持 1.0×，
證明這個量測反映的是「方向性」而非標註雜訊。

**2. 密集場景中，水平框的重疊多半是「幽靈重疊」。** 同類別相鄰物件中，
水平框 IoU ≥ 0.3 的配對裡，實際旋轉框 IoU < 0.1 的比例：

| 類別 | HBB IoU≥0.3 配對數 | 幽靈重疊率 |
|---|---:|---:|
| large vehicle | 810 | **100%** |
| ship | 736 | **99%** |
| harbor | 43 | **98%** |
| small vehicle | 42 | **93%** |

水平框偵測器的 NMS 會把這些視為重複偵測而誤殺 —— 停車場的卡車、
港口並排的船就這樣消失。旋轉框讓重疊度（與 NMS 的決策）回歸真實。

**3. 一圖勝千言** —— 同一個碼頭、同一份標註（5 組對照圖都在 `assets/`）：

![HBB vs OBB，碼頭 535 艘船](assets/hbb_vs_obb_1_P0706_ship.jpg)

## 部署 Benchmark：PyTorch vs ONNX Runtime vs TensorRT FP16

在 Colab **Tesla T4** 上把 fine-tuned `best.pt` 匯出成 ONNX 跟 TensorRT FP16 engine
（[notebooks/02_benchmark_colab.ipynb](notebooks/02_benchmark_colab.ipynb)，可獨立重現——
原計畫在本機 RTX 4090 build engine，實際本機硬體是 RTX 2070 8GB，因此改到 Colab 上做，
這同時也是「TensorRT 在 Windows 上安裝卡關」的替代方案）。batch=1、imgsz=1024，每個後端
20 次 warmup + 100 次計時取平均，engine 綁定這次 build 用的 GPU 型號與 TensorRT 版本。

**先做匯出精度驗證**（dota8 val，同工具同條件比較三個後端）：PyTorch mAP50=0.9950、
ONNX 0.9950、TensorRT 0.9950——三者沒有精度損失，符合這個專案一開始設的 <1 個百分點門檻
（YOLO26 的 end-to-end 匯出有已知 issue，ultralytics#23397 等）。

| 後端 | 檔案大小 (MB) | 平均延遲 | p50 | p95 | FPS |
|---|---:|---:|---:|---:|---:|
| PyTorch (FP32) | 48.7 | 71.54 ms | 71.49 ms | 74.08 ms | 14.0 |
| ONNX Runtime GPU | 85.3 | 79.09 ms | 79.67 ms | 81.43 ms | 12.6 |
| TensorRT FP16 | 45.0 | 20.22 ms | 20.14 ms | 21.09 ms | 49.4 |

**TensorRT FP16 比原生 PyTorch 快約 3.5 倍**，檔案也最小。**ONNX Runtime GPU 反而比原生
PyTorch 略慢**——沒有 TensorRT 這種圖編譯後端加持的話，ONNX Runtime 的 GPU 執行不一定會贏
PyTorch 自己的 cuDNN kernel；會保留這個後端是因為 Phase 6 的免費版 HF Space demo 是用它跑
CPU 推論，不是因為它是 GPU 上最快的選項。

拿到這組數字的過程踩了好幾輪 Colab 環境的坑，都不是這個專案自己程式碼的問題：`torch._dynamo`
內部版本兜不起來、HF 檔案 CDN 間歇性回傳簽章失效的下載連結、ONNX Runtime 的 CUDA 執行
provider 會讓整個 Colab 執行階段原生崩潰（後來改用獨立子行程隔離跑推論解決）。完整過程寫在
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)。

## 授權

- 程式碼：**AGPL-3.0**（[Ultralytics](https://github.com/ultralytics/ultralytics) 為 AGPL-3.0，fine-tune 權重屬衍生物、同授權）
- 資料集：**DOTA 限學術用途，禁止商業使用**

*（Demo GIF 與重現步驟於後續 Phase 補齊。English version: README.md）*
