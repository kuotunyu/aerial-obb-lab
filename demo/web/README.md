# Aerial OBB Lab Browser Demo

這是一個純 HTML／CSS／JavaScript 的 Browser-native OBB workbench。頁面開啟後會先顯示
repository 內的 Ultralytics 官方 `boats.jpg` 航拍範例；按下「開始 Detect」才會 lazy-load
固定版本 `onnxruntime-web@1.20.1` 的 JavaScript／WASM 與 repository 內的
privacy-sanitized YOLO26n-OBB derivative，並在目前的 browser session 執行 inference。
影像與模型不會上傳。

範例模型必須符合 `demo-model.json` 的固定 manifest。Browser 會限制 same-origin model URL、
下載大小與精確 byte length，並用 Web Crypto 驗證 SHA-256 digest 後才建立 ONNX session。
完成後可以在原圖與 detection result 之間切換、調整 confidence／class filter，並查看 rotated
polygons、provenance 與依 confidence 排序的完整數值表格；filter 與畫面切換不會重新執行模型。

BYOM 是預設收合的進階入口。使用者可自行選擇相容的 local YOLO26 OBB ONNX model 與影像；
兩個檔案都只在 browser session 中處理，並與官方範例共用 preprocessing、inference、decode、
filter 與 rendering pipeline。選入無效的新 model 不會取代最後一個已驗證的 session。

## 執行

在 repository root 啟動任一 static HTTP server：

```powershell
.venv/Scripts/python.exe -m http.server 8765 --directory demo/web
```

開啟 `http://localhost:8765`，確認原圖後按「開始 Detect」。首次載入或 browser cache miss 時，
固定版本的 runtime assets 會從 jsDelivr 取得，因此 Detect 是 non-zero-network；repository 內的
範例 model 與 image 則由同一 origin 提供。`ort.min.js` 使用 SHA-384 SRI 與 anonymous CORS；
這個 SRI 只涵蓋 JavaScript，不涵蓋 runtime 後續取得的 WASM assets。直接用 `file://` 開啟可能
被 browser 的 WASM／resource policy 阻擋。

## Model contract 與限制

<!-- claim:browser-scope -->
- Demo 與 BYOM model 的 input 必須是 `images [1,3,1024,1024]` float32 RGB CHW，output 必須符合
  end-to-end `output0 [1,N,7]`。
- BYOM 接受 user-supplied compatible model 與 image，兩者只在目前的 browser session 內處理。
- 這個 browser integration demo **does not represent** fine-tuned `yolo26m-obb` checkpoint 的
  accuracy、evaluation 或歷史 T4 latency；UI result 與 screenshot 也不是這些指標的 evidence。
<!-- /claim:browser-scope -->

- 支援 confidence threshold、15 類 class filter、rotated polygon rendering，以及完整數值表格。
- Manifest、runtime、model integrity、model contract、inference、output schema、render 或 image decode
  失敗時，UI 會清除 stale results、顯示固定且不洩漏 local filename 的錯誤，並保留安全的 retry／
  重新選擇路徑。Inference／output／render failure 會保留目前的 base image。
- 本路徑不需要 Python ML runtime、Torch、CUDA、DOTA、weights、HF token 或 secrets。

## 來源與授權

- 原始範例影像：[`https://ultralytics.com/images/boats.jpg`](https://ultralytics.com/images/boats.jpg)
- Model 與素材 provenance：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- Privacy transformation record：
  [`third_party/yolo26n-obb-privacy-sanitization.json`](third_party/yolo26n-obb-privacy-sanitization.json)
- Ultralytics model license：
  [`third_party/ULTRALYTICS-AGPL-3.0.txt`](third_party/ULTRALYTICS-AGPL-3.0.txt)

Repository code 為 **AGPL-3.0-or-later**。Bundled model 是 Ultralytics YOLO26n-OBB 的
privacy-sanitized AGPL derivative；使用者自行提供的 model 與影像仍受各自的 upstream software、
dataset、weight 與 image rights 約束。DOTA images／annotations 不包含在本 release。
