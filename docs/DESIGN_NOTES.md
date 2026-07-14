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
- ultralytics 自動下載的 DOTAv1（2GB）是**原始大圖**（800–20000px），必須 `split_trainval(rates=..., gap=500)` 切成 1024 重疊 tiles 才能訓練（官方作法）
- 切好的 tiles 快取到 HF **私有** dataset repo（DOTA 授權限學術用途，不公開重散布），Colab 斷線重跑先拉快取
- 官方推薦的多尺度組合是 `[0.5,1.0,1.5]`；實際訓練最後改用 `[0.8,1.2]`，理由見 D5

### D4. YOLO26 e2e 匯出的已知風險（2026-07-13）
- GitHub issues：#23397（ONNX 匯出後 NMS-free 行為丟失/精度掉）、#23645（FP16 匯出 output 仍是 FP32）、#24697（e2e ONNX 用法疑問）
- 對策：每次匯出後跑 val，與 PyTorch 權重差距 <1 mAP 點才視為通過

### D5. split_dota 尺度組合的取捨過程（2026-07-14）
Phase 2 正式訓練前，第一次用官方推薦的 `[0.5,1.0,1.5]` 三尺度在 A100(40GB) 上實測，一個
epoch 要 ~74 分鐘、60 epoch 換算 70+ 小時，不可行。診斷與調整過程（完整版見 T5）：
1. **先誤以為是 CPU dataloader 瓶頸**，改單一尺度 `[1.0]` 驗證：速度大幅改善（~15分/epoch），
   但完全犧牲了多尺度帶來的準確率增益
2. **發現 `split_dota` 的 tiles 數量跟 rate 的平方成正比**（不是線性）：crop window
   大小 = `1024/r`，r 越大 window 越小、切出的 tiles 越多。反推 `tiles(r) ≈ tiles(r=1.0) × r²`
3. 因此兩尺度 `[1.0, 1.5]` 幾乎跟三尺度一樣貴（`1.5` 這個尺度自己就佔掉原三尺度資料量六成
   以上），不是省時間的好選擇
4. 改用**窄範圍兩尺度 `[0.8, 1.2]`**：比官方的 `0.5/1.5` 溫和，仍同時保留「縮小看大範圍」
   （原意是想照顧 bridge/harbor 這類大型旋轉物件）與「放大看細節」（有利小物件）兩個方向，
   資料量壓在單一尺度的 ~2 倍，一個 epoch ~29 分鐘
5. **結果只驗證了一半的假設**：small vehicle/helicopter/soccer ball field 確實進步最多
   （符合「放大有利小物件」的預期），但 bridge/harbor 反而退步——完整數字見
   [training_results.md](training_results.md)。誠實記錄假設沒有完全應驗，比事後硬拗更有價值
- 另外 `batch` 實際跑的是 `40`，不是這輪討論定案的 `36`（notebook 上傳版本不一致），
  但完整跑完 28 epoch 沒有 OOM，等於意外驗證了 `batch=40` 在這個資料量下是安全的

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

### T4. 中文路徑下 OpenCV 讀寫圖片會靜默失敗（2026-07-13）
- 背景：專案最終路徑含中文與空格（`D:\CC_F5_專案\...\1_YOLO26 OBB 旋轉框偵測：訓練到部署`），是 Phase 0 原本刻意要避開的風險，後來使用者要求統一搬到這個慣用路徑下
- 症狀：`cv2.imread(path)` / `cv2.imwrite(path, img)` 在非 ASCII 路徑下不會拋例外，而是**靜默回傳 None / 寫入失敗**（OpenCV 底層用 ANSI codepage API 開檔）——比崩潰更危險，容易誤判「跑完了但其實沒寫到檔案」
- 解法：`src/obbkit/viz.py` 改用 `np.fromfile(path) + cv2.imdecode(...)` 讀、`cv2.imencode(...) + buf.tofile(path)` 寫，繞過 OpenCV 的路徑開檔，改用 Python/numpy 自己的檔案 I/O（原生支援 Unicode）。已在中文路徑下實測驗證通過
- 面試可講：這是「知道某函式庫在特定平台/編碼下有隱性限制」的經驗，以及用位元組流隔離掉外部 C 函式庫的路徑編碼問題是通用解法（不只 OpenCV，很多用 C/C++ 底層做檔案 I/O 的函式庫在 Windows 上都有同樣問題）

### T5. Colab A100 訓練速度異常慢，診斷出是 CPU 端瓶頸不是 GPU（2026-07-14）
- 症狀：`[0.5,1.0,1.5]` 三尺度、batch=16 在 A100(40GB) 上訓練，`GPU_mem` 用到 30.8G（幾乎
  滿）但吞吐量只有 ~1.8it/s，一個 epoch 要 74 分鐘，60 epoch 換算 70+ 小時
- 關鍵判斷依據：**VRAM 滿但吞吐量低**是 CPU 端 dataloader/OBB 標註增強跟不上 GPU 的典型
  訊號，不是算力或顯存不足——如果真的是算力瓶頸，加大 VRAM/batch 沒有意義，反而可能讓
  CPU 端更跟不上
- 修正動作：① 加上 CPU 核數自動偵測，`workers` 不超過實際核數避免超賣互搶資源 ② 換
  A100 80GB 高記憶體版本、`batch` 從 16 拉到 32-40（有更多 VRAM 餘裕才敢做這一步，且是在
  確認瓶頸主因是資料量而非算力後才加大）③ 資料量本身從三尺度砍到兩尺度窄範圍 `[0.8,1.2]`
  （見 D5）
- 效果：單張圖處理速度從 ~21 張/秒提升到 ~33-36 張/秒（batch 加大 + workers 修正的貢獻），
  疊加資料量減少後，一個 epoch 從 74 分鐘降到 ~29 分鐘；`results.csv` 的 `time` 欄位顯示
  28 epoch 實際訓練耗時 49899 秒（**~13.9 小時**），從最壞情況預估的 70+ 小時大幅壓縮
- 面試可講：這是「先用可觀測指標（VRAM 使用率 vs 吞吐量）縮小瓶頸範圍，而不是憑直覺調參」
  的診斷方法；也是一次「加大硬體資源前，先確認硬體不是瓶頸」的反直覺教訓——一開始想直接
  加大 batch/VRAM 解決慢的問題，但診斷後發現那個方向完全沒用，真正有效的槓桿是資料量與
  CPU 資源分配

## 面試 Q&A（Phase 7 收斂）

（草稿隨各 Phase 累積）

- **Q: Fine-tune 之後 mAP 幾乎沒變，是不是訓練失敗？**
  A: 不是。`yolo26m-obb.pt` 本身就是官方在 DOTAv1 上訓練到收斂的權重，這次是拿自己切的
  tiles 繼續訓練同一份資料，不是帶模型認識新領域，看到邊際效益趨近於零是預期中的結果。
  價值在於用同條件對照，誠實量化這個邊際效益，而不是製造一次好看的進步數字。
- **Q: 你怎麼判斷訓練慢是 CPU 瓶頸還是 GPU 瓶頸？**
  A: 看 VRAM 使用率跟吞吐量的組合——VRAM 幾乎打滿、但每秒處理的圖片數遠低於 GPU 該有的
  算力水準，代表 GPU 在等資料，不是資料在等 GPU。單看其中一個指標容易誤判。
- **Q: 為什麼多尺度切圖的資料量不是跟尺度數量成正比？**
  A: `split_dota` 的做法是把裁切視窗大小設成 `crop_size/r`，r 越大視窗越小，同一張大圖
  需要切的視窗數量跟 r² 成正比。所以 `1.5` 這種放大尺度切出的 tiles 遠多於 `0.5` 這種縮小
  尺度，資料量不是跟你選了幾個尺度線性相關，而是被最大的那個尺度主導。
