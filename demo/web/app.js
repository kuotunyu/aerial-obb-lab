// Browser-only YOLO26 OBB inference. Demo and BYOM share one state machine and pipeline.
const IMGSZ = 1024;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp";
const DEMO_PROVENANCE = "Ultralytics YOLO26n-OBB · privacy-sanitized AGPL derivative";
let ortPromise = null;

const ERROR_COPY = Object.freeze({
  DEMO_MANIFEST: "範例模型資訊無法驗證。請重新整理頁面，或使用 BYOM。",
  DEMO_MODEL_FETCH: "範例模型無法取得。請檢查網路後重試，或使用 BYOM。",
  DEMO_MODEL_SIZE: "範例模型大小驗證失敗。請重新整理頁面，或使用 BYOM。",
  DEMO_MODEL_DIGEST: "範例模型完整性驗證失敗。請重新整理頁面，或使用 BYOM。",
  DEMO_MODEL_URL: "範例模型來源驗證失敗。請重新整理頁面，或使用 BYOM。",
  RUNTIME_LOAD: "Browser runtime 無法載入。請檢查網路或 content blocker 後重試，或使用 BYOM。",
  MODEL_CONTRACT: "請選擇使用 images [1,3,1024,1024] 與 output0 [1,N,7] 的相容 ONNX。",
  IMAGE_DECODE: "Browser 無法解碼影像。請改選 PNG、JPEG 或 WebP。",
  INFERENCE_RUN: "推論未完成。請重試 Detect，或使用相容的 BYOM 模型。",
  OUTPUT_SCHEMA: "模型輸出不符合 output0 [1,N,7]。請改用相容的 end-to-end OBB export。",
  RENDER_RESULT: "結果無法呈現。請重新執行 Detect。",
});

const CLASS_NAMES = Object.freeze([
  "plane", "ship", "storage tank", "baseball diamond", "tennis court",
  "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
  "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
]);
const CLASS_COLORS = Object.freeze([
  "#075985", "#dc2626", "#b45309", "#7c3aed", "#0f766e",
  "#c2410c", "#0369a1", "#be123c", "#4338ca", "#15803d",
  "#a21caf", "#1d4ed8", "#ca8a04", "#c026d3", "#047857",
]);

const demoOriginalImage = document.getElementById("demoOriginalImage");
const demoFigure = document.getElementById("demoFigure");
const demoFigureLabel = document.getElementById("demoFigureLabel");
const demoDetectBtn = document.getElementById("demoDetectBtn");
const viewToggleBtn = document.getElementById("viewToggleBtn");
const resultControls = document.getElementById("resultControls");
const modelInput = document.getElementById("modelInput");
const modelLabel = document.getElementById("modelLabel");
const fileInput = document.getElementById("fileInput");
const fileDrop = document.getElementById("fileDrop");
const fileLabel = document.getElementById("fileLabel");
const confSlider = document.getElementById("confSlider");
const confVal = document.getElementById("confVal");
const classList = document.getElementById("classList");
const detectBtn = document.getElementById("detectBtn");
const runtimeRetryBtn = document.getElementById("runtimeRetryBtn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("canvas");
const canvasDescription = document.getElementById("canvasDescription");
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
  source: "demo",
  phase: "idle",
  generation: 0,
  session: null,
  sessionSource: null,
  image: null,
  cached: null,
  elapsedMs: null,
  manifest: null,
  demoModelBytes: null,
  view: "original",
};

CLASS_NAMES.forEach((name, index) => {
  const label = document.createElement("label");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.value = String(index);
  checkbox.name = "class-filter";
  checkbox.className = "class-cb";
  label.append(checkbox, document.createTextNode(name));
  classList.appendChild(label);
});

function setStatus(message, kind = "neutral") {
  statusEl.textContent = message;
  statusEl.dataset.kind = kind;
  runtimeRetryBtn.hidden = true;
}

function reportFailure(code) {
  const safe = Object.hasOwn(ERROR_COPY, code) ? code : "INFERENCE_RUN";
  clearResultPresentation();
  state.phase = "error";
  console.warn(`[AERIAL_OBB:${safe}]`);
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

function renderSummary(detections, elapsedMs = null) {
  summaryCount.textContent = String(detections.length);
  summaryTop.textContent = detections.length
    ? Math.max(...detections.map((detection) => detection.conf)).toFixed(3)
    : "—";
  runtimeValue.textContent = state.phase === "result" && Number.isFinite(elapsedMs)
    ? `${Math.round(elapsedMs)} ms`
    : "—";
}

function setInitialSummary() {
  summaryCount.textContent = "0";
  summaryTop.textContent = "—";
  runtimeValue.textContent = "—";
  modeBadge.textContent = "尚未 Detect";
  provenanceValue.textContent = "官方範例 · 尚未執行";
}

function resetResult() {
  state.cached = null;
  state.elapsedMs = null;
  state.view = "original";
  resultsBody.innerHTML = "";
  canvasDescription.textContent = "尚無 detection result。";
  canvasFrame.classList.remove("has-results");
  canvasFrame.hidden = true;
  resultControls.hidden = true;
  viewToggleBtn.hidden = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  renderSummary([]);
}

function clearResultPresentation() {
  resetResult();
  modeBadge.textContent = "NO RESULT";
  provenanceValue.textContent = "—";
}

function resetToDemoOriginal() {
  state.source = "demo";
  state.phase = "idle";
  state.image = demoOriginalImage.complete && demoOriginalImage.naturalWidth
    ? demoOriginalImage
    : null;
  resetResult();
  demoFigure.hidden = false;
  demoOriginalImage.hidden = false;
  demoFigureLabel.textContent = "原圖 · 尚未 Detect";
  demoDetectBtn.textContent = "開始 Detect";
  demoDetectBtn.disabled = state.image === null;
  setInitialSummary();
  setStatus(state.image ? "原圖已載入 · 尚未 Detect。" : "正在載入官方範例原圖…");
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
      if (!globalThis.ort) {
        ortPromise = null;
        reject(new Error("RUNTIME_LOAD"));
        return;
      }
      globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;
      globalThis.ort.env.wasm.numThreads = 1;
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

async function fetchDemoManifest() {
  let response;
  try {
    response = await fetch("demo-model.json", {
      cache: "force-cache",
      credentials: "same-origin",
      redirect: "error",
    });
    if (!response.ok) throw new Error("DEMO_MANIFEST");
    return DemoAssets.validateManifest(await response.json());
  } catch (_error) {
    throw new Error("DEMO_MANIFEST");
  }
}

function exactNames(actual, expected) {
  return Array.isArray(actual) && actual.length === expected.length &&
    expected.every((name, index) => actual[index] === name);
}

function metadataEntry(metadata, name, index) {
  if (Array.isArray(metadata)) return metadata[index];
  if (metadata && typeof metadata === "object") return metadata[name];
  return null;
}

function metadataShape(metadata) {
  return metadata?.shape || metadata?.dimensions || metadata?.dims || null;
}

function validateSessionContract(candidate) {
  if (!exactNames(candidate?.inputNames, ["images"]) ||
      !exactNames(candidate?.outputNames, ["output0"])) {
    throw new Error("MODEL_CONTRACT");
  }
  const input = metadataEntry(candidate.inputMetadata, "images", 0);
  const output = metadataEntry(candidate.outputMetadata, "output0", 0);
  if (input) {
    const shape = metadataShape(input);
    if (input.type !== "float32" || !exactNames(shape, [1, 3, 1024, 1024])) {
      throw new Error("MODEL_CONTRACT");
    }
  }
  if (output) {
    const shape = metadataShape(output);
    if (
      output.type !== "float32" ||
      !Array.isArray(shape) ||
      shape.length !== 3 ||
      shape[0] !== 1 ||
      !Number.isInteger(shape[1]) ||
      shape[1] <= 0 ||
      shape[2] !== 7
    ) {
      throw new Error("MODEL_CONTRACT");
    }
  }
}

async function assignCandidate(candidate, source, generation) {
  try {
    validateSessionContract(candidate);
  } catch (_error) {
    if (typeof candidate.release === "function") await candidate.release();
    throw new Error("MODEL_CONTRACT");
  }
  if (!isCurrentGeneration(generation)) {
    if (typeof candidate.release === "function") await candidate.release();
    return null;
  }
  const previous = state.session;
  state.session = candidate;
  state.sessionSource = source;
  if (previous && previous !== candidate && typeof previous.release === "function") {
    await previous.release();
  }
  return candidate;
}

async function ensureDemoSession(generation) {
  if (state.session && state.sessionSource === "demo") return state.session;
  const manifest = state.manifest || await fetchDemoManifest();
  if (!isCurrentGeneration(generation)) return null;
  state.manifest = manifest;
  const [runtime, modelBytes] = await Promise.all([
    loadOrtRuntime(),
    state.demoModelBytes
      ? Promise.resolve(state.demoModelBytes)
      : DemoAssets.fetchVerifiedModel(manifest),
  ]);
  if (!isCurrentGeneration(generation)) return null;
  state.demoModelBytes = modelBytes;
  let candidate;
  try {
    candidate = await runtime.InferenceSession.create(new Uint8Array(modelBytes), {
      executionProviders: ["wasm"],
    });
  } catch (_error) {
    throw new Error("MODEL_CONTRACT");
  }
  return assignCandidate(candidate, "demo", generation);
}

async function replaceByomSession(file, generation) {
  const runtime = await loadOrtRuntime();
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (!isCurrentGeneration(generation)) return null;
  let candidate;
  try {
    candidate = await runtime.InferenceSession.create(bytes, {executionProviders: ["wasm"]});
  } catch (_error) {
    throw new Error("MODEL_CONTRACT");
  }
  return assignCandidate(candidate, "byom", generation);
}

function preprocess(image) {
  const geometry = OBB.letterboxGeometry(image.naturalWidth, image.naturalHeight, IMGSZ);
  const offscreen = document.createElement("canvas");
  offscreen.width = IMGSZ;
  offscreen.height = IMGSZ;
  const offscreenContext = offscreen.getContext("2d");
  offscreenContext.fillStyle = "rgb(114,114,114)";
  offscreenContext.fillRect(0, 0, IMGSZ, IMGSZ);
  offscreenContext.drawImage(
    image, 0, 0, image.naturalWidth, image.naturalHeight,
    geometry.padX, geometry.padY, geometry.newWidth, geometry.newHeight,
  );
  const pixels = offscreenContext.getImageData(0, 0, IMGSZ, IMGSZ).data;
  return {chw: OBB.rgbaToChw(pixels), geometry};
}

function selectedClasses() {
  return new Set(
    Array.from(document.querySelectorAll(".class-cb:checked"), (checkbox) =>
      Number(checkbox.value)),
  );
}

function decodeCachedOutput() {
  if (!state.cached || !state.image) return [];
  const output = OBB.selectEndToEndOutput(state.cached.results);
  return OBB.decodeDetections(
    output,
    state.cached.geometry,
    Number(confSlider.value),
    selectedClasses(),
    CLASS_NAMES.length,
  );
}

function drawDetections(detections) {
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  ctx.drawImage(state.image, 0, 0);
  ctx.lineWidth = Math.max(3, canvas.width / 500);
  for (const detection of detections) {
    const corners = OBB.rotatedCorners(detection);
    ctx.strokeStyle = CLASS_COLORS[detection.cls % CLASS_COLORS.length];
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let index = 1; index < corners.length; index += 1) {
      ctx.lineTo(corners[index][0], corners[index][1]);
    }
    ctx.closePath();
    ctx.stroke();
  }
  canvasFrame.classList.add("has-results");
}

function fillTable(detections) {
  resultsBody.innerHTML = "";
  for (const detection of detections) {
    const row = document.createElement("tr");
    const angle = (detection.angle * 180) / Math.PI;
    const cells = [
      CLASS_NAMES[detection.cls],
      detection.conf.toFixed(3),
      detection.w.toFixed(1),
      detection.h.toFixed(1),
      angle.toFixed(1),
    ];
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    resultsBody.appendChild(row);
  }
}

function renderCanvasDescription(detections) {
  if (!detections.length) {
    canvasDescription.textContent = "目前篩選條件下沒有 detections；canvas 沒有 oriented polygons。";
    return;
  }
  canvasDescription.textContent = detections.map((detection) => {
    const angle = (detection.angle * 180) / Math.PI;
    return `class=${CLASS_NAMES[detection.cls]}; confidence=${detection.conf.toFixed(3)}; ` +
      `center-x=${detection.cx.toFixed(1)} px; center-y=${detection.cy.toFixed(1)} px; ` +
      `width=${detection.w.toFixed(1)} px; height=${detection.h.toFixed(1)} px; ` +
      `angle=${angle.toFixed(1)}°.`;
  }).join(" ");
}

function renderCachedOutput() {
  if (!state.cached || !state.image) return [];
  let detections;
  try {
    detections = decodeCachedOutput();
  } catch (_error) {
    reportFailure("OUTPUT_SCHEMA");
    return null;
  }
  try {
    fillTable(detections);
    renderSummary(detections, state.cached.elapsedMs);
    renderCanvasDescription(detections);
    if (state.view === "result") drawDetections(detections);
  } catch (_error) {
    reportFailure("RENDER_RESULT");
    return null;
  }
  return detections;
}

function setResultView(view) {
  if (!state.cached || (view !== "original" && view !== "result")) return;
  state.view = view;
  if (view === "original") {
    canvasFrame.hidden = true;
    demoFigure.hidden = false;
    demoOriginalImage.hidden = false;
    demoFigureLabel.textContent = "原圖";
    viewToggleBtn.textContent = "查看結果";
    return;
  }
  demoFigure.hidden = true;
  demoFigureLabel.textContent = "Detect 結果";
  canvasFrame.hidden = false;
  viewToggleBtn.textContent = "查看原圖";
  renderCachedOutput();
}

async function runActiveInference(source, generation) {
  const image = state.image;
  const session = state.session;
  if (!image || !session || state.sessionSource !== source) return null;
  state.source = source;
  state.phase = "loading";
  resetResult();
  demoDetectBtn.disabled = true;
  detectBtn.disabled = true;
  let prepared;
  let results;
  let started;
  try {
    setStatus("正在準備 1024px RGB CHW input…", "running");
    prepared = preprocess(image);
    const tensor = new globalThis.ort.Tensor("float32", prepared.chw, [1, 3, IMGSZ, IMGSZ]);
    setStatus("正在本機 browser 執行 inference…", "running");
    started = performance.now();
    results = await session.run({images: tensor});
    if (!isCurrentGeneration(generation)) return null;
  } catch (_error) {
    if (isCurrentGeneration(generation)) reportFailure("INFERENCE_RUN");
    return null;
  }
  const elapsedMs = performance.now() - started;
  try {
    OBB.selectEndToEndOutput(results);
  } catch (_error) {
    reportFailure("OUTPUT_SCHEMA");
    return null;
  }
  state.cached = {results, geometry: prepared.geometry, elapsedMs, source};
  state.elapsedMs = elapsedMs;
  state.phase = "result";
  modeBadge.textContent = source === "demo"
    ? "LOCAL BROWSER INFERENCE"
    : "BYOM · LOCAL BROWSER INFERENCE";
  provenanceValue.textContent = source === "demo" ? DEMO_PROVENANCE : "Local files";
  resultControls.hidden = false;
  viewToggleBtn.hidden = source !== "demo";
  setResultView("result");
  const detections = renderCachedOutput();
  if (detections === null) return null;
  if (source === "demo") demoDetectBtn.textContent = "再次 Detect";
  setStatus(
    detections.length
      ? `完成 · ${detections.length} 個 detections`
      : "完成 · 沒有符合條件的 detections",
    "success",
  );
  resultTitle.focus();
  return detections;
}

async function runDemo() {
  const generation = nextGeneration();
  state.source = "demo";
  state.image = demoOriginalImage;
  resetResult();
  setInitialSummary();
  demoFigure.hidden = false;
  demoFigureLabel.textContent = "原圖";
  demoDetectBtn.disabled = true;
  setStatus("正在驗證範例模型與 Browser runtime…", "running");
  try {
    const session = await ensureDemoSession(generation);
    if (!session || !isCurrentGeneration(generation)) return;
    await runActiveInference("demo", generation);
  } catch (error) {
    if (!isCurrentGeneration(generation)) return;
    reportFailure(error?.message || "MODEL_CONTRACT");
  } finally {
    if (isCurrentGeneration(generation)) demoDetectBtn.disabled = false;
  }
}

async function handleModelSelection(file) {
  if (!file) return;
  const generation = nextGeneration();
  state.source = "byom";
  resetResult();
  modeBadge.textContent = "NO RESULT";
  provenanceValue.textContent = "—";
  setStatus("正在載入 local ONNX model…", "running");
  try {
    const session = await replaceByomSession(file, generation);
    if (!session || !isCurrentGeneration(generation)) return;
    modelLabel.textContent = "Local ONNX model ready";
    setStatus(state.image ? "Model ready · 可以開始 BYOM Detect。" : "Model ready · 請選擇影像。", "success");
    detectBtn.disabled = !(state.image && state.sessionSource === "byom");
  } catch (error) {
    if (!isCurrentGeneration(generation)) return;
    reportFailure(error?.message === "RUNTIME_LOAD" ? "RUNTIME_LOAD" : "MODEL_CONTRACT");
  }
}

function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("IMAGE_DECODE"));
    image.src = url;
  });
}

async function loadImageFile(file) {
  const generation = nextGeneration();
  state.source = "byom";
  resetResult();
  setStatus("正在解碼 local image…", "running");
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImageUrl(url);
    if (!isCurrentGeneration(generation)) return;
    state.image = image;
    fileLabel.textContent = "Local image ready";
    detectBtn.disabled = !(state.session && state.sessionSource === "byom");
    setStatus(
      detectBtn.disabled ? "影像已載入 · 請選擇 ONNX model。" : "影像已載入 · 可以開始 BYOM Detect。",
      detectBtn.disabled ? "neutral" : "success",
    );
  } catch (_error) {
    if (isCurrentGeneration(generation)) reportFailure("IMAGE_DECODE");
  } finally {
    URL.revokeObjectURL(url);
  }
}

confSlider.addEventListener("input", () => {
  confVal.textContent = Number(confSlider.value).toFixed(2);
  renderCachedOutput();
});
classList.addEventListener("change", renderCachedOutput);
viewToggleBtn.addEventListener("click", () => {
  setResultView(state.view === "result" ? "original" : "result");
});
demoDetectBtn.addEventListener("click", () => void runDemo());
modelInput.addEventListener("change", () => void handleModelSelection(modelInput.files[0]));
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) void loadImageFile(fileInput.files[0]);
});
detectBtn.addEventListener("click", () => {
  if (state.image && state.sessionSource === "byom") {
    void runActiveInference("byom", nextGeneration());
  }
});
runtimeRetryBtn.addEventListener("click", () => {
  ortPromise = null;
  if (state.source === "demo") void runDemo();
  else if (modelInput.files[0]) void handleModelSelection(modelInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDrop.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  fileDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDrop.classList.remove("dragover");
  });
});
fileDrop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) void loadImageFile(file);
});

if (demoOriginalImage.complete && demoOriginalImage.naturalWidth) {
  resetToDemoOriginal();
} else {
  demoDetectBtn.disabled = true;
  demoOriginalImage.addEventListener("load", resetToDemoOriginal, {once: true});
  demoOriginalImage.addEventListener("error", () => reportFailure("IMAGE_DECODE"), {once: true});
}
