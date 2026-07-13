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

（隨實作補充）

## 面試 Q&A（Phase 7 收斂）

（草稿隨各 Phase 累積）
