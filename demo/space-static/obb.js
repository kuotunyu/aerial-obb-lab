// Pure preprocessing geometry and YOLO26 end-to-end OBB decoding.
// This file has no DOM or ONNX Runtime dependency so the exact browser contract can also run in Node.
(function expose(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OBB = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildApi() {
  "use strict";

  function finite(value, label) {
    if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
    return value;
  }

  function positive(value, label) {
    finite(value, label);
    if (value <= 0) throw new Error(`${label} must be positive`);
    return value;
  }

  function letterboxGeometry(width, height, size) {
    positive(width, "image width");
    positive(height, "image height");
    positive(size, "letterbox size");
    const scale = Math.min(size / width, size / height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    const padX = Math.floor((size - newWidth) / 2);
    const padY = Math.floor((size - newHeight) / 2);
    return { scale, newWidth, newHeight, padX, padY };
  }

  function rgbaToChw(rgba) {
    if (!rgba || typeof rgba.length !== "number" || rgba.length % 4 !== 0) {
      throw new Error("expected flat RGBA pixels");
    }
    const plane = rgba.length / 4;
    const chw = new Float32Array(plane * 3);
    for (let pixel = 0; pixel < plane; pixel++) {
      chw[pixel] = rgba[pixel * 4] / 255;
      chw[plane + pixel] = rgba[pixel * 4 + 1] / 255;
      chw[2 * plane + pixel] = rgba[pixel * 4 + 2] / 255;
    }
    return chw;
  }

  function selectEndToEndOutput(results) {
    if (!results || !Object.prototype.hasOwnProperty.call(results, "output0")) {
      throw new Error("expected ONNX output named output0");
    }
    const tensor = results.output0;
    if (
      !tensor ||
      !Array.isArray(tensor.dims) ||
      tensor.dims.length !== 3 ||
      tensor.dims[0] !== 1 ||
      !Number.isInteger(tensor.dims[1]) ||
      tensor.dims[1] <= 0 ||
      tensor.dims[2] !== 7
    ) {
      throw new Error("expected output0 shape [1,N,7]");
    }
    if (!tensor.data || tensor.data.length !== tensor.dims[1] * 7) {
      throw new Error("output0 data length does not match its dimensions");
    }
    return tensor.data;
  }

  function decodeDetections(output, geometry, confidence, classIds, classCount) {
    if (!output || typeof output.length !== "number" || output.length % 7 !== 0) {
      throw new Error("expected flattened [N,7] output");
    }
    const scale = positive(geometry && geometry.scale, "letterbox scale");
    const padX = finite(geometry.padX, "letterbox padX");
    const padY = finite(geometry.padY, "letterbox padY");
    finite(confidence, "confidence threshold");
    if (confidence < 0 || confidence > 1) throw new Error("confidence threshold must be in [0, 1]");
    if (!Number.isInteger(classCount) || classCount <= 0) {
      throw new Error("class count must be a positive integer");
    }
    if (!classIds || typeof classIds.has !== "function" || typeof classIds.size !== "number") {
      throw new Error("class filter must be a Set-like object");
    }

    const detections = [];
    for (let offset = 0; offset < output.length; offset += 7) {
      const row = Array.from(output.slice(offset, offset + 7));
      row.forEach((value, column) => finite(value, `output column ${column}`));
      const [cx, cy, width, height, score, classId, angle] = row;
      if (width <= 0 || height <= 0) throw new Error("box width and height must be positive");
      if (!Number.isInteger(classId) || classId < 0 || classId >= classCount) {
        throw new Error(`class id must be an integer in [0, ${classCount})`);
      }
      if (score < 0 || score > 1) throw new Error("output confidence must be in [0, 1]");
      if (score < confidence || (classIds.size > 0 && !classIds.has(classId))) continue;
      detections.push({
        cx: (cx - padX) / scale,
        cy: (cy - padY) / scale,
        w: width / scale,
        h: height / scale,
        conf: score,
        cls: classId,
        angle,
      });
    }
    detections.sort((left, right) => right.conf - left.conf);
    return detections;
  }

  function rotatedCorners(detection) {
    const cx = finite(detection && detection.cx, "box cx");
    const cy = finite(detection.cy, "box cy");
    const width = positive(detection.w, "box width");
    const height = positive(detection.h, "box height");
    const angle = finite(detection.angle, "box angle");
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const halfWidth = width / 2;
    const halfHeight = height / 2;
    return [
      [-halfWidth, -halfHeight],
      [halfWidth, -halfHeight],
      [halfWidth, halfHeight],
      [-halfWidth, halfHeight],
    ].map(([x, y]) => [cx + x * cos - y * sin, cy + x * sin + y * cos]);
  }

  return {
    letterboxGeometry,
    rgbaToChw,
    selectEndToEndOutput,
    decodeDetections,
    rotatedCorners,
  };
});
