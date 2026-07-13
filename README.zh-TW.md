# YOLO26 OBB × DOTA：航拍旋轉框偵測，從訓練到部署

> 🚧 進行中 — 完整計畫見 [docs/PLAN.md](docs/PLAN.md)。

以 **YOLO26-OBB**（旋轉框）在 **DOTAv1** 航拍資料集上 fine-tune，展示完整生命週期：
Colab A100 訓練 → 與官方 baseline 對照評估 → 用數據回答「為什麼航拍場景需要 OBB」→ ONNX / TensorRT FP16 匯出與三框架 latency benchmark → Gradio demo + Hugging Face Space。

## 授權

- 程式碼：**AGPL-3.0**（[Ultralytics](https://github.com/ultralytics/ultralytics) 為 AGPL-3.0，fine-tune 權重屬衍生物、同授權）
- 資料集：**DOTA 限學術用途，禁止商業使用**

*（完整結果表格、benchmark、HBB vs OBB 分析與重現步驟於 Phase 7 補齊。English version: README.md）*
