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
const BOX_COLOR = "#22c55e";

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
const ctx = canvas.getContext("2d");
const resultsBody = document.getElementById("resultsBody");

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

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateDetectEnabled() {
  detectBtn.disabled = !(session && currentImage);
}

async function releaseSession() {
  if (session && typeof session.release === "function") await session.release();
  session = null;
}

async function loadModelFile(file) {
  setStatus("loading local model...");
  const modelBytes = new Uint8Array(await file.arrayBuffer());
  const nextSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  await releaseSession();
  session = nextSession;
  modelLabel.textContent = file.name;
  setStatus("model ready");
  updateDetectEnabled();
}

modelInput.addEventListener("change", async () => {
  const file = modelInput.files[0];
  if (!file) return;
  detectBtn.disabled = true;
  try {
    await loadModelFile(file);
  } catch (e) {
    await releaseSession();
    modelLabel.textContent = "Choose a compatible .onnx model";
    setStatus("model load failed: " + e.message);
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
    setStatus(session ? "image loaded, click Detect" : "image loaded; choose a model");
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
  ctx.drawImage(currentImage, 0, 0);
  ctx.strokeStyle = BOX_COLOR;
  ctx.fillStyle = BOX_COLOR;
  ctx.lineWidth = Math.max(2, canvas.width / 500);
  ctx.font = `${Math.max(14, canvas.width / 100)}px sans-serif`;

  for (const d of dets) {
    const corners = OBB.rotatedCorners(d);

    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let k = 1; k < 4; k++) ctx.lineTo(corners[k][0], corners[k][1]);
    ctx.closePath();
    ctx.stroke();

    const label = `${CLASS_NAMES[d.cls]} ${d.conf.toFixed(2)}`;
    ctx.fillText(label, corners[0][0], Math.max(12, corners[0][1] - 4));
  }
}

function fillTable(dets) {
  resultsBody.innerHTML = "";
  for (const d of dets) {
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
    setStatus("preprocessing…");
    const { chw, geometry } = preprocess(currentImage);
    const tensor = new ort.Tensor("float32", chw, [1, 3, IMGSZ, IMGSZ]);

    setStatus("running inference in your browser…");
    const t0 = performance.now();
    const feeds = { images: tensor };
    const results = await session.run(feeds);
    const output = OBB.selectEndToEndOutput(results);
    const ms = (performance.now() - t0).toFixed(0);

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
    setStatus(`done in ${ms}ms — ${dets.length} detection(s)`);
  } catch (e) {
    console.error(e);
    setStatus("error: " + e.message);
  } finally {
    updateDetectEnabled();
  }
});
