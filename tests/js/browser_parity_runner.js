"use strict";

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "..");
const OBB = require(path.join(root, "demo", "space-static", "obb.js"));
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const geometry = OBB.letterboxGeometry(
  fixture.letterbox.width,
  fixture.letterbox.height,
  fixture.letterbox.target,
);
const chw = Array.from(OBB.rgbaToChw(Uint8ClampedArray.from(fixture.rgba.values)));
const decode = fixture.decode;
const detections = OBB.decodeDetections(
  Float32Array.from(decode.output),
  decode.geometry,
  decode.confidence,
  new Set(decode.class_ids),
  decode.class_count,
);
const corners = OBB.rotatedCorners(detections[0]);

const invalidErrors = {};
for (const scenario of fixture.invalid_outputs) {
  try {
    OBB.decodeDetections(
      Float32Array.from(scenario.values),
      decode.geometry,
      decode.confidence,
      new Set(),
      decode.class_count,
    );
    invalidErrors[scenario.name] = null;
  } catch (error) {
    invalidErrors[scenario.name] = error.message;
  }
}

process.stdout.write(JSON.stringify({ geometry, chw, detections, corners, invalidErrors }));
