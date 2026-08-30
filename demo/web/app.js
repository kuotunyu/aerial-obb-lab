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

const ERROR_COPY = Object.freeze({
  SHOWCASE_ASSET: "Synthetic fixture 無法載入。請重新整理頁面，或改用 BYOM。",
  RUNTIME_LOAD: "Browser runtime 無法載入。請檢查網路或 content blocker 後重試；Synthetic Showcase 仍可使用。",
  MODEL_CONTRACT: "請選擇使用 images [1,3,1024,1024] 與 output0 [1,N,7] 的相容 ONNX。",
  IMAGE_DECODE: "Browser 無法解碼影像。請改選 PNG、JPEG 或 WebP。",
  INFERENCE_RUN: "推論未完成。請確認模型 contract、重新選擇影像後再試。",
  OUTPUT_SCHEMA: "模型輸出不符合 output0 [1,N,7]。請改用相容的 end-to-end OBB export。",
  RENDER_RESULT: "結果無法呈現。請重新載入 Synthetic Showcase，或重新執行 Detect。",
});

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
const runtimeRetryBtn = document.getElementById("runtimeRetryBtn");
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
const MODEL_PROMPT_HTML = '選擇相容的 <code>.onnx</code> model';
const IMAGE_PROMPT = "選擇或拖放一張影像";

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
  runtimeRetryBtn.hidden = true;
}

function reportFailure(code) {
  const safe = Object.hasOwn(ERROR_COPY, code) ? code : "INFERENCE_RUN";
  if (["INFERENCE_RUN", "OUTPUT_SCHEMA", "RENDER_RESULT"].includes(safe)) {
    clearResultPresentation();
  }
  console.warn("[AERIAL_OBB:" + safe + "]");
  setStatus(ERROR_COPY[safe], "error");
  runtimeRetryBtn.hidden = safe !== "RUNTIME_LOAD";
}

function nextGeneration() {
  state.generation += 1;
  return state.generation;
}

function isCurrentGeneration(token) {
  return token === state.generation;
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

function resetByomReadiness() {
  modelInput.value = "";
  fileInput.value = "";
  modelLabel.innerHTML = MODEL_PROMPT_HTML;
  fileLabel.textContent = IMAGE_PROMPT;
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

function clearResultPresentation() {
  canvasFrame.classList.remove("has-results");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) ctx.drawImage(state.image, 0, 0);
  resetResult();
}

function decodeCachedOutput() {
  if (!state.cached || !state.image) return [];
  const classes = new Set(
    Array.from(document.querySelectorAll(".class-cb:checked")).map((cb) => Number(cb.value))
  );
  const output = OBB.selectEndToEndOutput(state.cached.results);
  const dets = OBB.decodeDetections(
    output, state.cached.geometry, Number(confSlider.value), classes, CLASS_NAMES.length
  );
  return dets;
}

function renderCachedOutput() {
  if (!state.cached || !state.image) return [];
  let dets;
  try {
    dets = decodeCachedOutput();
  } catch (_error) {
    reportFailure("OUTPUT_SCHEMA");
    return null;
  }
  try {
    drawDetections(dets);
    fillTable(dets);
    renderSummary(dets, state.cached?.elapsedMs ?? null);
  } catch (_error) {
    reportFailure("RENDER_RESULT");
    return null;
  }
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
  if (!isCurrentGeneration(generation)) return false;
  const modelBytes = new Uint8Array(await file.arrayBuffer());
  if (!isCurrentGeneration(generation)) return false;
  const candidate = await runtime.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  if (!isCurrentGeneration(generation)) {
    if (typeof candidate.release === "function") await candidate.release();
    if (!isCurrentGeneration(generation)) return false;
  }

  try {
    validateSessionContract(candidate);
  } catch (_error) {
    if (typeof candidate.release === "function") await candidate.release();
    if (!isCurrentGeneration(generation)) return false;
    throw new Error("MODEL_CONTRACT");
  }

  const previous = state.session;
  state.session = candidate;
  modelLabel.textContent = "Local ONNX model ready";
  if (previous && typeof previous.release === "function") {
    await previous.release();
    if (!isCurrentGeneration(generation)) return false;
  }
  return true;
}

async function handleModelSelection(file) {
  if (!file) return;
  const generation = nextGeneration();
  clearSyntheticResult();
  detectBtn.disabled = true;
  setStatus("正在載入 local ONNX model…", "running");
  try {
    const assigned = await replaceModelSession(file, generation);
    if (!isCurrentGeneration(generation) || !assigned) return;
    setStatus("Model ready · 請選擇影像。", "success");
    updateDetectEnabled();
  } catch (error) {
    if (!isCurrentGeneration(generation)) return;
    reportFailure(error?.message === "RUNTIME_LOAD" ? "RUNTIME_LOAD" : "MODEL_CONTRACT");
    updateDetectEnabled();
  }
}

modelInput.addEventListener("change", () => {
  void handleModelSelection(modelInput.files[0]);
});

runtimeRetryBtn.addEventListener("click", () => {
  void handleModelSelection(modelInput.files[0]);
});

async function loadImageFile(file) {
  const generation = nextGeneration();
  clearSyntheticResult();
  state.image = null;
  updateDetectEnabled();
  setStatus("正在解碼 local image…", "running");
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImageUrl(url);
    if (!isCurrentGeneration(generation)) return;
    state.image = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    updateDetectEnabled();
    fileLabel.textContent = "Local image ready";
    setStatus(
      state.session ? "影像已載入 · 可以開始 Detect。" : "影像已載入 · 請選擇 ONNX model。",
      state.session ? "success" : "neutral",
    );
  } catch (_error) {
    if (!isCurrentGeneration(generation)) return;
    reportFailure("IMAGE_DECODE");
  } finally {
    URL.revokeObjectURL(url);
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) void loadImageFile(fileInput.files[0]);
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
  if (file) void loadImageFile(file);
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
  const generation = nextGeneration();
  const image = state.image;
  const session = state.session;
  detectBtn.disabled = true;
  try {
    let geometry;
    let results;
    let t0;
    try {
      setStatus("正在準備 1024px RGB CHW input…", "running");
      const prepared = preprocess(image);
      geometry = prepared.geometry;
      const tensor = new ort.Tensor("float32", prepared.chw, [1, 3, IMGSZ, IMGSZ]);

      setStatus("正在本機 browser 執行 inference…", "running");
      t0 = performance.now();
      const feeds = { images: tensor };
      results = await session.run(feeds);
      if (!isCurrentGeneration(generation)) return;
    } catch (_error) {
      if (!isCurrentGeneration(generation)) return;
      reportFailure("INFERENCE_RUN");
      return;
    }
    const elapsedMs = performance.now() - t0;
    state.cached = { results, geometry, provenance: "Local files", elapsedMs };
    state.elapsedMs = elapsedMs;
    state.phase = "result";
    const dets = renderCachedOutput();
    if (dets === null) return;
    setStatus(
      dets.length ? `完成 · ${dets.length} 個 detections` : "完成 · 沒有符合條件的 detections",
      "success",
    );
  } finally {
    if (isCurrentGeneration(generation)) updateDetectEnabled();
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
  const generation = nextGeneration();
  state.phase = "loading";
  detectBtn.disabled = true;
  resetByomReadiness();
  setStatus("正在載入 Synthetic Showcase…", "running");
  try {
    await releaseSession();
    if (!isCurrentGeneration(generation)) return;
    state.image = null;
    canvasFrame.classList.remove("has-results");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resetResult();
    const image = await loadImageUrl(OBB_SHOWCASE.imageUrl);
    if (!isCurrentGeneration(generation)) return;
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
    if (renderCachedOutput() === null) return;
    runtimeValue.textContent = "N/A · no inference";
    resultTitle.focus();
    setStatus("Synthetic fixture 已載入 · 沒有執行模型推論。", "success");
  } catch (_error) {
    if (!isCurrentGeneration(generation)) return;
    state.phase = "error";
    reportFailure("SHOWCASE_ASSET");
  }
}

showcaseBtn.addEventListener("click", () => {
  void activateShowcase();
});
