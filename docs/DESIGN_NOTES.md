# DESIGN_NOTES — 技術決策、踩坑記錄、面試 Q&A

> 隨實作進度持續補充。面試 Q&A 在 Phase 7 收斂成 8–10 題。

## 決策記錄

### D1. 模型選 yolo26m-obb（2026-07-13）
- 官方 DOTAv1 test 指標（imgsz 1024）：n=78.9/52.4、s=80.9/54.8、**m=81.0/55.3**、l=81.6/56.2、x=81.7/56.7（mAP50/mAP50-95）
- 選 m：A100 40GB 以 imgsz 1024 訓練綽綽有餘；比 l/x 輕，Colab T4 benchmark 與本機 RTX 2070 8GB demo 都跑得動；比 s 的 mAP50-95 高 0.5
- 被淘汰的替代：l/x（部署端太重、對作品集敘事無增益）、s（保留為 VRAM/時間爆炸時的退路）
- YOLO26 vs YOLO11：YOLO26 是當前最新一代，end-to-end NMS-free、去 DFL、MuSGD optimizer，官方宣稱 DOTA OBB 比 YOLO11 高最多 +3.4 mAP

### D2. 訓練上 Colab A100、benchmark 上 Colab T4（2026-07-13）
- 本機實際是 RTX 2070 8GB（手冊假設 4090 不符）、Windows 10、驅動 551.23
- TensorRT engine build 與三框架 benchmark 改在 Colab（Linux 上 TensorRT 以 pip 安裝順暢；T4 同為 Turing 世代、官方 speed 基準也用 T4）
- 這同時就是「TensorRT 在 Windows 卡關」的替代方案：交付可重現的 Colab notebook 而非綁死本機環境

### D3. DOTAv1 需要 split_dota 前處理（2026-07-13）
- ultralytics 自動下載的 DOTAv1（2GB）是**原始大圖**（800–20000px），必須 `split_trainval(rates=[0.5,1.0,1.5], gap=500)` 切成 1024 重疊 tiles 才能訓練（官方作法）
- 切好的 tiles 快取到 HF **私有** dataset repo（DOTA 授權限學術用途，不公開重散布），Colab 斷線重跑先拉快取

### D4. YOLO26 e2e 匯出的已知風險（2026-07-13）
- GitHub issues：#23397（ONNX 匯出後 NMS-free 行為丟失/精度掉）、#23645（FP16 匯出 output 仍是 FP32）、#24697（e2e ONNX 用法疑問）
- 對策：每次匯出後跑 val，與 PyTorch 權重差距 <1 mAP 點才視為通過

## 踩坑記錄

### T1. torch ≥2.9 在 Windows 上 `WinError 1114`（c10.dll 初始化失敗）（2026-07-13）
- 症狀：`import torch` 直接炸 `OSError: [WinError 1114]`，逐一 ctypes 載入定位到 `c10.dll` 本身 init 失敗（相依 DLL 都正常）
- 原因：torch 2.9.0 起 Windows wheel 需要較新的 MSVC++ Redistributable（≥14.50；本機是 14.44）— [pytorch/pytorch#169429](https://github.com/pytorch/pytorch/issues/169429)
- 更新 VC++ redist 需要管理員權限（winget 卡 UAC），改走免權限路線：**本機 torch pin `>=2.6,<2.9`**（2.8.x + cu128 在 Win10 正常）；Colab 端不受影響用最新
- 面試可講：診斷手法（逐 DLL ctypes 載入縮小範圍）+ 環境隔離決策（本機開發環境 vs 雲端訓練環境各自鎖版本）

### T2. onnxruntime 1.23+ 同樣 DLL init 失敗、onnxslim 原生崩潰（2026-07-13）
- 症狀一：`import onnxruntime`（1.27、1.23.2 都試過）→ `DLL load failed while importing onnxruntime_pybind11_state`（同 T1 的 MSVC 執行階段太舊家族）；**1.20.1 驗證可用** → pin `>=1.20,<1.21`
- 症狀二：ultralytics 匯出 ONNX 時 onnxslim 0.1.94 直接 access violation（0xC0000005）讓 Python 整個死掉（exit code -1073741819）→ 本機匯出一律 `simplify=False`；Colab（Linux）維持預設 slimming
- 附帶發現：ultralytics 匯出時會自動安裝 `onnxruntime-gpu`，與 CPU 版並存可能引發 DLL 衝突，uv sync 會把它清掉（venv 由 uv 管理的好處）
- 根治方式：更新 MSVC++ Redistributable 到 14.50+（需管理員權限）後即可解除所有 pin；目前的 pin 是「無管理員權限也能完整跑通」的工程取捨

### T3. uv hardlink 快取被 pip 污染 → 「重裝也修不好」的假象（2026-07-13）
- 症狀：`onnxruntime==1.20.1` 第一次驗證可用，之後同版本怎麼重裝（含整目錄刪除重灌）都 DLL init 失敗
- 原因鏈：ultralytics 匯出時用 pip 自動安裝 `onnxruntime-gpu`（與 CPU 版共用 `onnxruntime/` 目錄）→ 檔案操作寫穿了 uv 的 **hardlink** → 污染 uv 快取本體 → 之後每次 uv 安裝都從髒快取 hardlink 回來
- 解法：`uv cache clean onnxruntime` 強制重新下載 + 設 `UV_LINK_MODE=copy` 杜絕再發
- 教訓：uv 管理的 venv 裡混用 pip 有風險；ONNX 推論驗證固定 `device="cpu"` 讓 ultralytics 不觸發 `onnxruntime-gpu` 自動安裝

## 面試 Q&A（Phase 7 收斂）

（草稿隨各 Phase 累積）
