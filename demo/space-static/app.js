// Fully client-side YOLO26n-OBB inference via ONNX Runtime Web.
// I/O format verified empirically against the ultralytics Python reference (see
// docs/DESIGN_NOTES.md): input "images" [1,3,1024,1024] float32 /255, letterboxed
// (scale=min(1024/W,1024/H), pad 114 gray, centered); output "output0" [1,300,7] =
// [cx, cy, w, h, conf, cls, angle_rad] per detection in letterboxed pixel space.

const IMGSZ = 1024;
const MODEL_URL = "yolo26n-obb.onnx";
const CLASS_NAMES = [
  "plane", "ship", "storage tank", "baseball diamond", "tennis court",
  "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
  "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
];
const BOX_COLOR = "#22c55e";

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

async function ensureSession() {
  if (session) return session;
  setStatus("loading model (~10MB, first time only)…");
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
  });
  setStatus("model ready");
  return session;
}
// kick off model loading in the background as soon as the page is ready, so the
// first click on "Detect" doesn't have to wait for the full download.
ensureSession().catch((e) => setStatus("model load failed: " + e.message));

function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    detectBtn.disabled = false;
    fileLabel.textContent = file.name;
    resultsBody.innerHTML = "";
    setStatus("image loaded, click Detect");
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
  const scale = Math.min(IMGSZ / W, IMGSZ / H);
  const newW = Math.round(W * scale);
  const newH = Math.round(H * scale);
  const padX = Math.floor((IMGSZ - newW) / 2);
  const padY = Math.floor((IMGSZ - newH) / 2);

  const off = document.createElement("canvas");
  off.width = IMGSZ;
  off.height = IMGSZ;
  const octx = off.getContext("2d");
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, IMGSZ, IMGSZ);
  octx.drawImage(img, 0, 0, W, H, padX, padY, newW, newH);

  const { data } = octx.getImageData(0, 0, IMGSZ, IMGSZ); // RGBA, HWC, uint8
  const chw = new Float32Array(3 * IMGSZ * IMGSZ);
  const plane = IMGSZ * IMGSZ;
  for (let p = 0; p < plane; p++) {
    chw[p] = data[p * 4] / 255; // R
    chw[plane + p] = data[p * 4 + 1] / 255; // G
    chw[2 * plane + p] = data[p * 4 + 2] / 255; // B
  }
  return { chw, scale, padX, padY };
}

// Decode raw [300,7] output into detections in original-image pixel space.
function postprocess(output, scale, padX, padY, confThresh, classMask) {
  const dets = [];
  const n = output.length / 7;
  for (let i = 0; i < n; i++) {
    const o = i * 7;
    const conf = output[o + 4];
    if (conf < confThresh) continue;
    const cls = Math.round(output[o + 5]);
    if (classMask.size > 0 && !classMask.has(cls)) continue;
    dets.push({
      cx: (output[o] - padX) / scale,
      cy: (output[o + 1] - padY) / scale,
      w: output[o + 2] / scale,
      h: output[o + 3] / scale,
      conf,
      cls,
      angle: output[o + 6], // radians
    });
  }
  dets.sort((a, b) => b.conf - a.conf);
  return dets;
}

function drawDetections(dets) {
  ctx.drawImage(currentImage, 0, 0);
  ctx.strokeStyle = BOX_COLOR;
  ctx.fillStyle = BOX_COLOR;
  ctx.lineWidth = Math.max(2, canvas.width / 500);
  ctx.font = `${Math.max(14, canvas.width / 100)}px sans-serif`;

  for (const d of dets) {
    const cos = Math.cos(d.angle);
    const sin = Math.sin(d.angle);
    const hw = d.w / 2;
    const hh = d.h / 2;
    // corners in local frame, rotated into image space, translated to (cx,cy)
    const corners = [
      [-hw, -hh],
      [hw, -hh],
      [hw, hh],
      [-hw, hh],
    ].map(([x, y]) => [d.cx + x * cos - y * sin, d.cy + x * sin + y * cos]);

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
  if (!currentImage) return;
  detectBtn.disabled = true;
  try {
    setStatus("preprocessing…");
    const { chw, scale, padX, padY } = preprocess(currentImage);
    const tensor = new ort.Tensor("float32", chw, [1, 3, IMGSZ, IMGSZ]);

    setStatus("running inference in your browser…");
    const t0 = performance.now();
    const sess = await ensureSession();
    const feeds = { images: tensor };
    const results = await sess.run(feeds);
    const output = results[Object.keys(results)[0]].data;
    const ms = (performance.now() - t0).toFixed(0);

    const confThresh = Number(confSlider.value);
    const classMask = new Set(
      Array.from(document.querySelectorAll(".class-cb:checked")).map((cb) => Number(cb.value))
    );
    const dets = postprocess(output, scale, padX, padY, confThresh, classMask);

    drawDetections(dets);
    fillTable(dets);
    setStatus(`done in ${ms}ms — ${dets.length} detection(s)`);
  } catch (e) {
    console.error(e);
    setStatus("error: " + e.message);
  } finally {
    detectBtn.disabled = false;
  }
});
