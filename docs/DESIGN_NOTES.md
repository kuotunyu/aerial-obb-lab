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

### T6. `uv run` 在中文路徑下重寫 editable-install `.pth` 檔，讓整個 venv 開不了機（2026-07-14）
- 症狀：`uv run --group demo python demo/app.py` 炸 `Fatal Python error: init_import_site`
  → `UnicodeDecodeError: 'cp950' codec can't decode byte 0x88`，而且**炸完之後連原本能跑的
  `.venv/Scripts/python.exe` 直接執行都一起壞掉**——不是單一指令失敗，是整個 venv 被寫壞了
- 原因鏈：`uv run` 每次執行前會重新同步 editable install，重寫
  `site-packages/_editable_impl_yolo26_dota_obb.pth`（內容是本專案的絕對路徑，含中文字，
  UTF-8 編碼寫入）→ Python 內建 `site.py` 處理 `.pth` 檔案時用系統預設 codepage
  （這台機器是 cp950 繁體中文）解碼，不是 UTF-8 → 路徑裡的中文字節在 cp950 下不是合法序列
  → 直接讓 Python **啟動階段**（比使用者程式碼還早）就死掉，任何後續用同一個 venv 啟動的
  Python 都會炸，不限於觸發的那個指令
- 解法：刪掉那個壞掉的 `.pth` 檔案（本專案沒有任何程式碼真的依賴 editable install，
  `demo/app.py`、`scripts/*.py` 都自己手動 `sys.path.insert` 加 `src/`），之後**不要用
  `uv run`**、改直接呼叫 `.venv/Scripts/python.exe` 啟動（`.claude/launch.json` 已改成
  這樣），繞開會重寫 `.pth` 的路徑
- 面試可講：這是 T4（OpenCV 中文路徑靜默失敗）的姊妹踩坑，但更嚴重——不是單一函式庫的
  問題，是 **Python 直譯器啟動流程本身**在中文/非 ASCII 路徑下會炸，而且是「一次寫壞、
  之後每次啟動都壞」的延遲效應，比當下就報錯更難追（一開始以為是 `uv` 的問題，實際上是
  它間接觸發了 CPython `site.py` 的既有限制）

### T7. Colab 預裝 torch 內部版本兜不起來（2026-07-14）
- 症狀：`model.val()` 觸發 `torch._dynamo.config` 被 import，`TypeError: Config() got an
  unexpected keyword argument 'deprecated'`——torch 自己的 `_dynamo/config.py` 呼叫
  `Config(deprecated=True,...)`，但它引用的 `Config` 類別定義不支援這個參數
- 這是 torch 這個 package 自己兩個內部檔案版本兜不起來，不是我們的程式碼或 ultralytics 的
  問題；Phase 2 訓練那次用同一台 Colab 沒遇到，研判是 Colab base image 之後悄悄換了一個
  有問題的 torch build
- 解法：安裝 ultralytics 之前先 `%pip install -q -U torch` 明確重裝一個乾淨穩定版
- 面試可講：不要預設「雲端環境每次都一樣」——Colab 的預裝套件版本會隨時間變動，
  可重現的 notebook 應該明確控管關鍵依賴版本，而不是依賴當下環境剛好是什麼

### T8. HF Hub 檔案下載連結的 CDN 簽章間歇性失效（2026-07-14）
- 症狀：`hf_hub_download` 下載 `best.pt` 連續 6 次（換過有無 `local_dir`、換過重試邏輯）都
  撞到同一個 `403 Forbidden ... SignatureError: invalid key pair id`，卡在同一個
  `us.gcp.cdn.hf.co/xet-bridge-us/...` 網址——是 HF 那個檔案在 Xet CDN 儲存後端的簽章
  基礎設施出問題，不是下載程式碼寫錯（也曾遇過另一種症狀：進度卡在 97% 不動，同一個根因）
- 這種問題從客戶端角度沒辦法修：換下載函式的參數（`local_dir` 有無）、重試、關閉 Xet
  加速通道都試過，都是同一個網址、同一個簽章錯誤
- 解法：不依賴會出問題的雲端下載，改成**手動把本機已有的檔案上傳進 Colab**（用 Colab 檔案
  總管拖曳），notebook 邏輯改成「本機檔案優先，找不到才嘗試 HF 下載」
- 面試可講：當一個外部服務的失敗模式在你這端完全無法透過改寫用戶端程式碼修復時（同樣輸入、
  同樣錯誤、換了兩種完全不同的程式碼路徑結果一樣），要有判斷力儘早放棄在同一個依賴上重試，
  改找繞過整個依賴的替代路徑，而不是無止盡地嘗試各種客戶端參數組合

### T9. ONNX Runtime 的 CUDA 執行 provider 原生崩潰，拖垮整個 Colab 執行階段（2026-07-14）
- 症狀：log 印到 `Using ONNX Runtime ... with CUDAExecutionProvider` 後，沒有任何 Python
  traceback，Colab 直接斷線——這是 C++ 層級的原生崩潰（很可能是這個環境 CUDA 13 跟
  onnxruntime-gpu/TensorRT 的版本沒有完全兜好），`try/except` 完全接不住，因為死掉的是
  整個 process，不是拋出一個可以攔截的例外
- 解法：把 ONNX / TensorRT 的推論（parity 檢查跟 benchmark 都要）丟到**獨立子行程**執行
  （`subprocess.run` 呼叫一個獨立的 worker script），子行程崩潰只會讓那次 `subprocess.run`
  回傳非 0 或逾時，可以在主行程裡偵測到、標記那個後端「crashed」、繼續完成其他後端的量測，
  不會讓整個 notebook 跟 Colab 執行階段一起死掉
- 順便處理了「訓練套件安裝過程要求重啟執行階段，把 Python 變數清空但磁碟檔案還在」的狀況：
  把 `best.pt`/`best.onnx`/`best.engine` 三個匯出步驟都改成「產出檔案已存在就跳過」，讓
  「全部執行」不管重跑幾次都安全、不會重做已完成的工作，也不用整個刪除執行階段重來
- 面試可講：這是「防禦不可信賴的原生依賴」的標準模式——當一個函式庫可能整個拖垮 process
  （而不是丟出可攔截的例外）時，唯一可靠的隔離手段是行程邊界（subprocess/多行程），
  Python 語言層級的 `try/except` 對這類失敗是無效的；另外也是「讓長流程可以安全重跑」
  （idempotent pipeline）的實例——用「產出物已存在就跳過」取代「假設每次都從零開始」，
  讓不穩定的雲端環境下的長流程變得可以放心重試

### T10. 測試 `demo/space/app.py` 時誤判「卡住」——其實是自己把 Gradio 伺服器啟動了（2026-07-15）
- 症狀：用 `import app` 直接呼叫函式測試（跟測 `demo/app.py` 一樣的手法），process 一直沒
  反應，看起來像卡死；用工作管理員查發現該 process 累積 CPU 時間極低（跑了好幾分鐘只有
  ~11 秒 CPU time），代表它不是在算東西、是在等待
- 原因：`demo/space/app.py` 最後一行 `demo.launch()`**沒有包在** `if __name__ == "__main__":`
  裡面（`demo/app.py` 有包）——這是刻意的正確設計，因為 HF Space 部署時就是直接執行這支
  script，不是被 import。但這代表本機用 `import app` 的方式測試，import 這個動作本身就會
  觸發 `launch()`，啟動一個會一直跑下去等 HTTP 請求的 Gradio 伺服器，不會「執行完畢返回」
- 解法：測試前先把 `gr.Blocks.launch` monkey-patch 成空函式再 import，讓 import 能正常
  跑完不觸發真的啟動伺服器，之後就能直接呼叫 `detect()` 驗證邏輯；另外也直接開瀏覽器連
  到那個意外啟動的伺服器確認 UI 渲染正常，兩種驗證方式都做了
- 面試可講：診斷「這個 process 是真的卡住還是在正常等待」，看累積 CPU 時間比看「還在跑」
  更準——卡死或死鎖通常伴隨接近 0 的 CPU 使用率或忽然衝到 100%，正常等待 I/O（像這裡等
  HTTP 請求）CPU 時間會維持極低但穩定；另外也是「兩個檔案表面相似但關鍵一行差異」會造成
  完全不同行為的例子，測試前該先看程式碼而不是照搬另一支腳本的測試手法

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
- **Q: ONNX Runtime GPU 為什麼比原生 PyTorch 慢？TensorRT 又為什麼快這麼多？**
  A: ONNX Runtime 的 CUDA 執行 provider 基本上是逐算子（op-by-op）呼叫 cuDNN/cuBLAS，
  跟 PyTorch eager mode 做的事情差不多，沒有本質上的優勢，遇到某些算子排程/記憶體配置沒
  PyTorch 調得好時甚至會更慢。TensorRT 的優勢是做**圖編譯**：把整個計算圖攤開、做算子融合
  （operator fusion）、選擇針對這張 GPU 實測最快的 kernel、轉 FP16，這些是 ONNX Runtime
  單純換個 runtime 執行不會做的最佳化，所以 TensorRT 才會有數倍差距，不是「換個 runtime
  就變快」這麼簡單。
- **Q: 遇到一個會讓整個 process 崩潰、`try/except` 接不住的依賴，你怎麼處理？**
  A: 語言層級的例外處理只能接住「拋出例外」這種失敗模式，接不住原生崩潰（segfault、C++
  層級的 fatal error）這種會直接終止整個 process 的失敗。唯一可靠的隔離手段是**行程邊界**
  ——把不信任的程式碼丟到子行程執行，父行程只需要檢查子行程的 exit code 跟輸出，子行程死
  掉不會連累父行程。這次拿 ONNX Runtime GPU 在 Colab 上的原生崩潰當例子，實作了一個最小
  的 subprocess worker 模式來隔離它。
