// Fully client-side bring-your-own-model YOLO26 OBB inference via ONNX Runtime Web.
// I/O format is enforced by the synthetic browser parity fixture in tests/fixtures:
// input "images" [1,3,1024,1024] float32 /255, letterboxed
// (scale=min(1024/W,1024/H), pad 114 gray, centered); output "output0" [1,300,7] =
// [cx, cy, w, h, conf, cls, angle_rad] per detection in letterboxed pixel space.

const IMGSZ = 1024;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp";
let ortPromise = null;

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
const showcaseBtn = document.getElementById("showcaseBtn");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const canvasFrame = document.getElementById("canvasFrame");
const ctx = canvas.getContext("2d");
const resultsBody = document.getElementById("resultsBody");
const summaryCount = document.getElementById("summaryCount");
const summaryTop = document.getElementById("summaryTop");
const runtimeValue = document.getElementById("runtimeValue");
const modeBadge = document.getElementById("modeBadge");
const provenanceValue = document.getElementById("provenanceValue");
const resultTitle = document.getElementById("resultTitle");

const state = {
  mode: "none", phase: "idle", generation: 0,
  session: null, image: null, cached: null, elapsedMs: null,
};

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
  renderCachedOutput();
});
classList.addEventListener("change", renderCachedOutput);

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
  detectBtn.disabled = !(state.session && state.image);
}

function loadOrtRuntime() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  if (ortPromise) return ortPromise;
  ortPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = ORT_URL;
    script.integrity = ORT_INTEGRITY;
    script.crossOrigin = "anonymous";
    script.onload = () => {
      globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;
      resolve(globalThis.ort);
    };
    script.onerror = () => {
      script.remove();
      ortPromise = null;
      reject(new Error("RUNTIME_LOAD"));
    };
    document.head.appendChild(script);
  });
  return ortPromise;
}

async function releaseSession() {
  const current = state.session;
  state.session = null;
  if (current && typeof current.release === "function") await current.release();
}

function resetResult() {
  state.cached = null;
  state.elapsedMs = null;
  resultsBody.innerHTML = "";
  renderSummary([]);
  modeBadge.textContent = "NO RESULT";
  provenanceValue.textContent = "—";
}

function renderCachedOutput() {
  if (!state.cached || !state.image) return [];
  const classes = new Set(
    Array.from(document.querySelectorAll(".class-cb:checked")).map((cb) => Number(cb.value))
  );
  const output = OBB.selectEndToEndOutput(state.cached.results);
  const dets = OBB.decodeDetections(
    output, state.cached.geometry, Number(confSlider.value), classes, CLASS_NAMES.length
  );
  drawDetections(dets);
  fillTable(dets);
  renderSummary(dets, state.cached.elapsedMs);
  return dets;
}

function clearSyntheticResult() {
  if (state.mode === "synthetic") state.image = null;
  state.mode = "byom";
  state.phase = "idle";
  state.cached = null;
  state.elapsedMs = null;
  canvasFrame.classList.remove("has-results");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resetResult();
}

function validateSessionContract(candidate) {
  if (
    !candidate?.inputNames?.includes("images") ||
    !candidate?.outputNames?.includes("output0")
  ) {
    throw new Error("SESSION_CONTRACT");
  }
}

async function replaceModelSession(file, generation) {
  const runtime = await loadOrtRuntime();
  const modelBytes = new Uint8Array(await file.arrayBuffer());
  const candidate = await runtime.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });

  try {
    validateSessionContract(candidate);
  } catch (_error) {
    if (typeof candidate.release === "function") await candidate.release();
    throw new Error("SESSION_CONTRACT");
  }
  if (generation !== state.generation) {
    if (typeof candidate.release === "function") await candidate.release();
    return false;
  }

  const previous = state.session;
  state.session = candidate;
  modelLabel.textContent = "Local ONNX model ready";
  if (previous && typeof previous.release === "function") await previous.release();
  return true;
}

modelInput.addEventListener("change", async () => {
  const file = modelInput.files[0];
  if (!file) return;
  const generation = ++state.generation;
  clearSyntheticResult();
  detectBtn.disabled = true;
  setStatus("正在載入 local ONNX model…", "running");
  try {
    const assigned = await replaceModelSession(file, generation);
    if (!assigned) return;
    setStatus("Model ready · 請選擇影像。", "success");
    updateDetectEnabled();
  } catch (_error) {
    if (generation !== state.generation) return;
    console.warn("Model load failed: incompatible local ONNX input.");
    setStatus("模型載入失敗，請確認 ONNX 格式與 output contract。", "error");
    updateDetectEnabled();
  }
});

function loadImageFile(file) {
  clearSyntheticResult();
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    state.image = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    updateDetectEnabled();
    fileLabel.textContent = file.name;
    setStatus(
      state.session ? "影像已載入 · 可以開始 Detect。" : "影像已載入 · 請選擇 ONNX model。",
      state.session ? "success" : "neutral",
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
  ctx.drawImage(state.image, 0, 0);
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
  if (!state.image || !state.session) return;
  detectBtn.disabled = true;
  try {
    setStatus("正在準備 1024px RGB CHW input…", "running");
    const { chw, geometry } = preprocess(state.image);
    const tensor = new ort.Tensor("float32", chw, [1, 3, IMGSZ, IMGSZ]);

    setStatus("正在本機 browser 執行 inference…", "running");
    const t0 = performance.now();
    const feeds = { images: tensor };
    const results = await state.session.run(feeds);
    const elapsedMs = performance.now() - t0;
    state.cached = { results, geometry, provenance: "Local files", elapsedMs };
    state.elapsedMs = elapsedMs;
    state.phase = "result";
    const dets = renderCachedOutput();
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

function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("SHOWCASE_ASSET"));
    image.src = url;
  });
}

async function activateShowcase() {
  const generation = ++state.generation;
  state.phase = "loading";
  await releaseSession();
  resetResult();
  const image = await loadImageUrl(OBB_SHOWCASE.imageUrl);
  if (generation !== state.generation) return;
  state.mode = "synthetic";
  state.phase = "result";
  state.image = image;
  state.cached = {
    results: OBB_SHOWCASE.results,
    geometry: OBB.letterboxGeometry(400, 200, 1024),
    provenance: OBB_SHOWCASE.provenance,
    elapsedMs: null,
  };
  modeBadge.textContent = "SYNTHETIC FIXTURE · NO INFERENCE";
  provenanceValue.textContent = OBB_SHOWCASE.provenance;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  renderCachedOutput();
  runtimeValue.textContent = "N/A · no inference";
  resultTitle.focus();
  setStatus("Synthetic fixture 已載入 · 沒有執行模型推論。", "success");
}

showcaseBtn.addEventListener("click", activateShowcase);
