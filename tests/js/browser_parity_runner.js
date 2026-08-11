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

const schema = fixture.output_schema;
const validOutput = OBB.selectEndToEndOutput({
  [schema.name]: {
    dims: schema.dims,
    data: new Float32Array(schema.expected_length),
  },
});
const schemaErrors = {};
for (const scenario of schema.invalid) {
  const results = Object.fromEntries(
    Object.entries(scenario.results).map(([name, tensor]) => [
      name,
      { dims: tensor.dims, data: new Float32Array(tensor.data_length) },
    ]),
  );
  try {
    OBB.selectEndToEndOutput(results);
    schemaErrors[scenario.name] = null;
  } catch (error) {
    schemaErrors[scenario.name] = error.message;
  }
}

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

process.stdout.write(JSON.stringify({
  geometry,
  chw,
  detections,
  corners,
  validOutputLength: validOutput.length,
  schemaErrors,
  invalidErrors,
}));
