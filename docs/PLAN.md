# 專案 1：YOLO26 OBB 旋轉框偵測 — 訓練到部署（實作計畫）

> 本檔為實作藍圖。實作開始後會複製到專案 repo 的 `docs/PLAN.md` 作為工作文件，過程中隨時可討論調整。

## Context（為什麼做、環境事實、已確認的決定）

求職作品集三專案中的第一個（依據《Claude-Code作品集執行手冊-v2》+ 專案 prompt）：用 YOLO26 在 DOTA 航拍資料集上展示 OBB 完整生命週期 —— 訓練（Colab A100）→ 評估 → 「為什麼需要 OBB」量化分析 → 部署匯出與 benchmark → Gradio demo 與 HF Space。

**實測環境（與手冊假設不同，已與使用者確認的調整）：**
- 本機是 Windows 10 + **RTX 2070 8GB**（非 4090），驅動 551.23 偏舊（最高支援 CUDA 12.4）
- C 槽只剩 27GB → 專案搬到 D 槽（D 槽 104GB 可用）；資料集與 HF 快取也放 D 槽
- **專案最終位置：`D:\CC_F5_專案\YOLO_OBB專案\1_YOLO26 OBB 旋轉框偵測：訓練到部署`**（2026-07-13 從 `D:\portfolio\yolo26-dota-obb` 再搬一次，統一到使用者慣用的 `D:\CC_F5_專案` 結構；含中文與空格 —— 已把 `src/obbkit/viz.py` 的 cv2 讀寫改成 `np.fromfile`+`imdecode` / `imencode`+`tofile` 位元組流方式，避開 OpenCV 對非 ASCII 路徑的靜默失敗風險，已在此路徑下驗證通過）
- **部署 benchmark（ONNX / TensorRT）改在 Colab GPU 上做**（使用者已選定），本機只負責開發、smoke test、Gradio demo
- **暫不推 GitHub**（使用者之後自行處理）：git 只做本機 init + 每階段 commit；Colab notebook 設計成自包含、由使用者手動上傳到 Colab
- 全域 CLAUDE.md 使用者自理，不在本專案處理
- 權重/成果交換走 HF Hub（使用者已有 write token）

**動工前查證結論（2026-07-13，官方文件）：**
- **YOLO26 OBB 已正式釋出**：`yolo26{n,s,m,l,x}-obb.pt`，DOTAv1 **test split** 官方指標：n=78.9/52.4、s=80.9/54.8、**m=81.0/55.3**、l=81.6/56.2、x=81.7/56.7（mAP50 / mAP50-95，imgsz 1024）
- ultralytics 最新版 **8.4.93**（2026-07-12 發布），支援 Python 3.8–3.13
- **DOTAv1 自動下載（2GB）是原始大圖**（800–20000px，train 1411 / val 458 / test 937 張，15 類），訓練前必須用 `ultralytics.data.split_dota.split_trainval(rates=[0.5,1.0,1.5], gap=500)` 切成 1024 重疊 tiles（官方作法）
- **DOTA8**（`dota8.yaml`，約 1MB，8 張切好的圖）自動下載，適合本機 smoke test
- DOTA 授權：**限學術用途、禁商用**；Ultralytics 為 AGPL-3.0（衍生權重照此標註）
- 已知坑：YOLO26 end-to-end（NMS-free）匯出 ONNX 有 issue（#23397 精度/重複框、#23645 FP16 輸出型別、#24697 e2e 用法）→ 匯出後必須做精度驗證

**模型選型：`yolo26m-obb`（維持預設，理由寫進 DESIGN_NOTES）**
m 級 21.2M params / 183 GFLOPs：A100 40GB 以 imgsz 1024 訓練綽綽有餘；相對 l/x，m 在 Colab T4 benchmark 與本機 2070 8GB demo 都跑得動；相對 s 只差 0.1 mAP50 但 mAP50-95 高 0.5。若訓練時 VRAM/時間有問題再降 s（決策點在 Phase 2）。

---

## 目錄結構（Phase 0 建立）

```
D:\portfolio\yolo26-dota-obb\
├── src\obbkit\            # 可重用程式碼（分析、繪圖、HF 上傳 callback）
├── scripts\               # smoke_test、資料準備、匯出、分析入口
├── notebooks\             # 01_train_dotav1_a100.ipynb、02_benchmark_colab.ipynb
├── demo\                  # app.py（本機 Gradio）；space\（HF Space CPU 版，未部署）；
│                          # space-static\（實際部署的版本，瀏覽器端 ONNX Runtime Web）
├── assets\                # README 用圖（HBB vs OBB 對照、demo 截圖、訓練曲線）
├── docs\                  # PLAN.md（本檔）、DESIGN_NOTES.md
├── README.md / README.zh-TW.md / LICENSE(AGPL-3.0) / .gitignore
└── pyproject.toml         # uv 管理，鎖版本
```

資料與快取（不進 repo）：`D:\datasets`（ultralytics settings `datasets_dir`）、`D:\hf-cache`（`HF_HOME`）。

---

## 階段規劃

### Phase 0：搬家與環境（本機，~0.5 天）
1. 建 `D:\portfolio\yolo26-dota-obb`、git init、腳手架、.gitignore（排除權重/資料集/runs）
2. uv 建 venv（Python 3.11）+ `ultralytics==8.4.93` + CUDA 版 torch
   - **驅動 551.23 太舊裝不了新 torch CUDA wheel 時**：請使用者更新 NVIDIA 驅動（建議路線）；過渡備案 pin `torch 2.6 + cu124`
3. `yolo settings datasets_dir=D:\datasets`；設 `HF_HOME=D:\hf-cache`；檢查 `huggingface_hub` 登入（未登入引導使用者貼 token）
4. 驗證：`torch.cuda.is_available()`；`yolo26n-obb.pt` 對官方 boats.jpg 推論出旋轉框
5. 本計畫落地為 `docs/PLAN.md`；DESIGN_NOTES.md 開檔
6. **使用者動作**：之後把 Claude Code 開在新資料夾繼續

### Phase 1：DOTA8 smoke test（本機 2070，~0.5 天）
把整條路在極小規模跑通，全綠才准上 Colab：
1. train：`yolo26n-obb.pt` + `dota8.yaml`（epochs≈5、imgsz 1024、小 batch 配合 8GB）
2. val + predict：確認 metrics 與旋轉框輸出正常
3. **模擬斷線 resume**：中途停掉 → 從 `last.pt` `resume=True` 續訓成功
4. **HF push callback 預演**：`on_model_save` callback 把 checkpoint 傳上 HF 測試 repo（Phase 2 notebook 直接重用這段程式碼）
5. export ONNX 冒煙 + 用 `YOLO("*.onnx")` 推論一次
- 產出：`scripts/smoke_test.py`（含各步 OK/FAIL 總結表）

### Phase 2：DOTAv1 正式訓練 notebook（Colab A100，掛機數小時）
`notebooks/01_train_dotav1_a100.ipynb`，**自包含**（不依賴 GitHub）：
1. `pip install ultralytics==8.4.93`；Colab Secrets 讀 `HF_TOKEN` 登入
2. 資料：DOTAv1 自動下載 → `split_trainval(rates=[0.5,1.0,1.5], gap=500)` → **切好的資料打包上傳 HF 私有 dataset repo 作快取**（斷線重跑先嘗試從 HF 拉，免重切；私有存放以符合 DOTA 授權）
3. 訓練：`yolo26m-obb.pt` 起手 fine-tune，imgsz=1024、epochs=60（patience 早停）、AMP 預設開、batch 依 A100 40GB 設定；**A100 / T4 兩組 preset（預設 A100）**
4. **checkpoint 續傳**：`on_model_save` callback 定期 push `last.pt`/`best.pt`/`results.csv` 到 HF model repo；notebook 開頭偵測 HF 上有 `last.pt` → 自動下載 `resume=True`
5. 訓練完自動：val（fine-tuned）+ **同條件 val 官方 `yolo26m-obb.pt` 當 baseline** + 上傳 `best.pt` 與 metrics 到 HF
- **使用者操作**：上傳 notebook → A100 → 全部執行 → 貼回最後的指標輸出（跑完記得中斷執行階段）

### Phase 3：評估表（數字來自 Phase 2，本機寫 README）
README 對照表三行：官方 test-split 指標（引用文件）／官方權重在 **val split** 實測（我們的 baseline）／fine-tuned 在 val split。方法論誠實聲明：官方數字是 test split，可直接比較的是後兩行（同 split 同工具同條件）。**檢核點**：若官方權重 val 分數異常高（訓練可能含 val split），在 README 註明並改以此為「上限參考」解讀。

### Phase 4：「為什麼需要 OBB」量化分析（本機，面試重點）
只需 DOTAv1 **val 原圖+標註**（已在 D:\datasets）：
1. `src/obbkit/analysis.py`：每物件「軸對齊外接框面積 ÷ 旋轉框面積」= **面積膨脹率**，按 15 類統計（預期 ship / large-vehicle / harbor / bridge 最誇張）
2. 密集場景（harbor 船、停車場 small/large-vehicle）：相鄰物件 **HBB 兩兩 IoU vs OBB 兩兩 IoU** 分佈 → 證明 HBB 在密集斜向目標下互相重疊、NMS 會誤殺
3. 挑 5 張典型圖產出 HBB vs OBB 並排對照圖 → `assets/`
4. 寫成 README 獨立一節（數字 + 圖）

### Phase 5：部署匯出與 benchmark（Colab GPU — 已確認不在本機做）
`notebooks/02_benchmark_colab.ipynb`（**建議 T4**：Turing 架構、省運算單元、官方 speed 數字也是 T4 基準；附 A100 選項）：
1. 從 HF 拉 `best.pt` → export **ONNX** 與 **TensorRT FP16 engine**（Linux 上 pip 裝 TensorRT 順暢）
2. **匯出精度驗證**：exported 模型 val mAP 與 PyTorch 差距 <1pt 才算過（防 e2e 匯出已知 issue；必要時依官方 end2end 指南調整匯出參數）
3. benchmark：PyTorch / ONNX Runtime GPU / TensorRT FP16 的 latency 與 FPS（batch=1、imgsz=1024、warmup+多次取平均；標注 GPU 型號、註明 engine 綁定 GPU）→ README 表格
4. ONNX 上傳 HF（engine 綁 GPU，只留數據不發布檔案）
- README 註明：原計畫本機 4090 build engine，實際硬體為 2070 8GB，故改以 Colab GPU 交付（這就是「TensorRT 在 Windows 卡關的替代方案」）

### Phase 6：Gradio demo（本機）＋ HF Space CPU 版
1. `demo/app.py`（本機 2070）：上傳航拍圖 → 旋轉框視覺化（`Results.plot()` 原生支援 OBB）、confidence slider、類別過濾、每目標角度（`result.obb.xywhr`）與類別/信心度清單
2. `demo/space/`：**yolo26n-obb 匯出 ONNX + ultralytics 以 onnxruntime CPU 推論**（`YOLO("yolo26n-obb.onnx")`，避免手寫 OBB 後處理）；requirements.txt 鎖版本（torch CPU wheel）；Space card
3. **對外發布前確認**：HF model repo / Space 命名與公開設定，屆時再跟使用者確認一次才上傳

### Phase 7：收尾
1. README.md（英文）+ README.zh-TW.md：簡介、結果表、benchmark 表、OBB 分析節、demo GIF、本機+Colab 重現步驟、授權節（**DOTA 學術用途禁商用、Ultralytics AGPL-3.0、衍生權重 AGPL-3.0**）
2. `docs/DESIGN_NOTES.md`：選型理由（m vs s/l）、split_dota 多尺度切圖、e2e 匯出的坑、resume/checkpoint 設計、HBB vs OBB 分析方法、8–10 題面試 Q&A
3. HF：`best.pt` + ONNX + 中英雙語 model card；Space 部署並實測能開
4. 每階段 git commit（本機）；GitHub push 留給使用者
5. 交付：HF 連結清單 + 中英文各 3 句專案介紹（履歷用）

---

## 風險與備案
| 風險 | 對策 |
|---|---|
| 本機驅動 551.23 裝不了新 torch CUDA | 請使用者更新驅動（建議）；備案 pin torch 2.6+cu124 |
| Colab 斷線 | checkpoint 已定期 push HF，notebook 自動 resume（Phase 1 先驗證過） |
| split_dota 每次重切耗時 | 切好的資料快取到 HF 私有 dataset repo |
| e2e ONNX/TRT 匯出精度異常（已知 issues） | 匯出後 val 驗證 <1pt 差距；不過就按官方 end2end 指南改匯出參數或改傳統輸出 |
| A100 搶不到 | notebook 內建 T4 preset（降 batch），或換時段 |
| 官方權重 val 分數含水分（可能見過 val） | Phase 3 檢核點，README 誠實註明 |
| 子代理額度上限（今晨 5:20 重置） | 不影響主線逐步實作 |

## 驗證方式（每階段的「完成」定義）
- **P0**：cuda 可用、OBB 推論出旋轉框、HF 登入 OK
- **P1**：smoke test 腳本輸出全綠（train/val/predict/resume/HF push/ONNX 五項）
- **P2**：使用者貼回 Colab 指標，`best.pt` 出現在 HF repo
- **P3–P4**：README 表格與 5 張對照圖成品
- **P5**：三框架 benchmark 表 + 匯出精度驗證通過
- **P6**：本機 `python demo/app.py` 開起來可互動；Space 連結點開真的能跑
- **P7**：對照手冊 Part 6 檢查清單逐項打勾（GitHub 項除外，留給使用者）

## 實作時再確認的事項
- HF 帳號 username 與 repo/Space 命名（首次上傳前）
- 正式訓練 epochs 依第一次訓練曲線調整
- 若 m 級在 A100 時間/VRAM 不合理 → 降 s 並記錄理由
