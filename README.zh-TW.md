# YOLO26 OBB × DOTA：航拍旋轉框偵測，從訓練到部署

> 🚧 進行中 — 完整計畫見 [docs/PLAN.md](docs/PLAN.md)。

以 **YOLO26-OBB**（旋轉框）在 **DOTAv1** 航拍資料集上 fine-tune，展示完整生命週期：
Colab A100 訓練 → 與官方 baseline 對照評估 → 用數據回答「為什麼航拍場景需要 OBB」→ ONNX / TensorRT FP16 匯出與三框架 latency benchmark → Gradio demo + Hugging Face Space。

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

## 授權

- 程式碼：**AGPL-3.0**（[Ultralytics](https://github.com/ultralytics/ultralytics) 為 AGPL-3.0，fine-tune 權重屬衍生物、同授權）
- 資料集：**DOTA 限學術用途，禁止商業使用**

*（完整結果表格、benchmark、HBB vs OBB 分析與重現步驟於 Phase 7 補齊。English version: README.md）*
