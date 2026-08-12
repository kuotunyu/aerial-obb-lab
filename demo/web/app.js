// Fully client-side bring-your-own-model YOLO26 OBB inference via ONNX Runtime Web.
// I/O format is enforced by the synthetic browser parity fixture in tests/fixtures:
// input "images" [1,3,1024,1024] float32 /255, letterboxed
// (scale=min(1024/W,1024/H), pad 114 gray, centered); output "output0" [1,300,7] =
// [cx, cy, w, h, conf, cls, angle_rad] per detection in letterboxed pixel space.

const IMGSZ = 1024;
const CLASS_NAMES = [
  "plane", "ship", "storage tank", "baseball diamond", "tennis court",
  "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
  "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
];
const CLASS_COLORS = [
  "#50e3c2", "#4aa8ff", "#ffca5c", "#ff7a8a", "#a78bfa",
  "#34d399", "#f59e0b", "#38bdf8", "#fb7185", "#c084fc",
  "#2dd4bf", "#60a5fa", "#facc15", "#f472b6", "#8b5cf6",
];

const modelInput = document.getElementById("modelInput");
const modelLabel = document.getElementById("modelLabel");
const fileInput = document.getElementById("fileInput");
const fileDrop = document.getElementById("fileDrop");
const fileLabel = document.getElementById("fileLabel");
const confSlider = document.getElementById("confSlider");
const confVal = document.getElementById("confVal");
const classList = document.getElementById("classList");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const canvasFrame = document.getElementById("canvasFrame");
const ctx = canvas.getContext("2d");
const resultsBody = document.getElementById("resultsBody");
const summaryCount = document.getElementById("summaryCount");
const summaryTop = document.getElementById("summaryTop");
const runtimeValue = document.getElementById("runtimeValue");

let session = null;
let currentImage = null; // HTMLImageElement

CLASS_NAMES.forEach((name, i) => {
  const label = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.value = String(i);
  cb.className = "class-cb";
  label.appendChild(cb);
  label.appendChild(document.createTextNode(name));
  classList.appendChild(label);
});

confSlider.addEventListener("input", () => {
  confVal.textContent = Number(confSlider.value).toFixed(2);
});

function setStatus(message, kind = "neutral") {
  statusEl.textContent = message;
  statusEl.dataset.kind = kind;
}

function renderSummary(dets, elapsedMs = null) {
  summaryCount.textContent = String(dets.length);
  summaryTop.textContent = dets.length
    ? Math.max(...dets.map((d) => d.conf)).toFixed(3)
    : "—";
  runtimeValue.textContent = elapsedMs === null ? "—" : `${Math.round(elapsedMs)} ms`;
}

function updateDetectEnabled() {
  detectBtn.disabled = !(session && currentImage);
}

async function releaseSession() {
  if (session && typeof session.release === "function") await session.release();
  session = null;
}

async function loadModelFile(file) {
  setStatus("正在載入 local ONNX model…", "running");
  const modelBytes = new Uint8Array(await file.arrayBuffer());
  const nextSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  await releaseSession();
  session = nextSession;
  modelLabel.textContent = file.name;
  setStatus("Model ready · 請選擇影像。", "success");
  updateDetectEnabled();
}

modelInput.addEventListener("change", async () => {
  const file = modelInput.files[0];
  if (!file) return;
  detectBtn.disabled = true;
  try {
    await loadModelFile(file);
  } catch (_error) {
    await releaseSession();
    console.warn("Model load failed: incompatible local ONNX input.");
    modelLabel.textContent = "選擇相容的 .onnx model";
    setStatus("模型載入失敗，請確認 ONNX 格式與 output contract。", "error");
    updateDetectEnabled();
  }
});

function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    updateDetectEnabled();
    fileLabel.textContent = file.name;
    resultsBody.innerHTML = "";
    renderSummary([]);
    setStatus(
      session ? "影像已載入 · 可以開始 Detect。" : "影像已載入 · 請選擇 ONNX model。",
      session ? "success" : "neutral",
    );
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadImageFile(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((evt) =>
  fileDrop.addEventListener(evt, (e) => {
    e.preventDefault();
    fileDrop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  fileDrop.addEventListener(evt, (e) => {
    e.preventDefault();
    fileDrop.classList.remove("dragover");
  })
);
fileDrop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) loadImageFile(file);
});

// Letterbox `img` into an IMGSZxIMGSZ canvas, gray-padded, centered.
// Returns { blob: Float32Array (CHW, 1x3xIMGSZxIMGSZ), scale, padX, padY }.
function preprocess(img) {
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const geometry = OBB.letterboxGeometry(W, H, IMGSZ);

  const off = document.createElement("canvas");
  off.width = IMGSZ;
  off.height = IMGSZ;
  const octx = off.getContext("2d");
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, IMGSZ, IMGSZ);
  octx.drawImage(
    img,
    0,
    0,
    W,
    H,
    geometry.padX,
    geometry.padY,
    geometry.newWidth,
    geometry.newHeight,
  );

  const { data } = octx.getImageData(0, 0, IMGSZ, IMGSZ); // RGBA, HWC, uint8
  return { chw: OBB.rgbaToChw(data), geometry };
}

function drawDetections(dets) {
  canvasFrame.classList.remove("has-results");
  ctx.drawImage(currentImage, 0, 0);
  ctx.lineWidth = Math.max(2, canvas.width / 500);

  for (const d of dets) {
    const corners = OBB.rotatedCorners(d);
    ctx.strokeStyle = CLASS_COLORS[d.cls % CLASS_COLORS.length];

    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let k = 1; k < 4; k++) ctx.lineTo(corners[k][0], corners[k][1]);
    ctx.closePath();
    ctx.stroke();
  }
  void canvasFrame.offsetWidth;
  canvasFrame.classList.add("has-results");
}

function fillTable(dets) {
  resultsBody.innerHTML = "";
  const sorted = [...dets].sort((a, b) => b.conf - a.conf);
  for (const d of sorted) {
    const tr = document.createElement("tr");
    const deg = (d.angle * 180) / Math.PI;
    tr.innerHTML = `<td>${CLASS_NAMES[d.cls]}</td><td>${d.conf.toFixed(3)}</td>` +
      `<td>${d.w.toFixed(1)}</td><td>${d.h.toFixed(1)}</td><td>${deg.toFixed(1)}</td>`;
    resultsBody.appendChild(tr);
  }
}

detectBtn.addEventListener("click", async () => {
  if (!currentImage || !session) return;
  detectBtn.disabled = true;
  try {
    setStatus("正在準備 1024px RGB CHW input…", "running");
    const { chw, geometry } = preprocess(currentImage);
    const tensor = new ort.Tensor("float32", chw, [1, 3, IMGSZ, IMGSZ]);

    setStatus("正在本機 browser 執行 inference…", "running");
    const t0 = performance.now();
    const feeds = { images: tensor };
    const results = await session.run(feeds);
    const output = OBB.selectEndToEndOutput(results);
    const elapsedMs = performance.now() - t0;

    const confThresh = Number(confSlider.value);
    const classMask = new Set(
      Array.from(document.querySelectorAll(".class-cb:checked")).map((cb) => Number(cb.value))
    );
    const dets = OBB.decodeDetections(
      output,
      geometry,
      confThresh,
      classMask,
      CLASS_NAMES.length,
    );

    drawDetections(dets);
    fillTable(dets);
    renderSummary(dets, elapsedMs);
    setStatus(
      dets.length ? `完成 · ${dets.length} 個 detections` : "完成 · 沒有符合條件的 detections",
      "success",
    );
  } catch (_error) {
    console.warn("Inference failed: incompatible model input or output contract.");
    setStatus("Inference 失敗，請確認模型使用 images input 與 output0 [1,N,7]。", "error");
  } finally {
    updateDetectEnabled();
  }
});
