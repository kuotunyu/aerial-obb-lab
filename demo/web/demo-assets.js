(function exposeDemoAssets(root) {
  "use strict";

  const MAX_MODEL_BYTES = 15 * 1024 * 1024;
  const EXPECTED_CLASSES = Object.freeze([
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
  ]);
  const IMAGE_SOURCE = ["https:", "", "ultralytics.com", "images", "boats.jpg"].join("/");
  const MODEL_SOURCE = [
    "https:", "", "github.com", "ultralytics", "assets", "releases", "download",
    "v8.4.0", "yolo26n-obb.onnx",
  ].join("/");
  const EXPECTED = Object.freeze({
    schemaVersion: 1,
    id: "ultralytics-yolo26n-obb-demo",
    defaultConfidence: 0.25,
    classes: EXPECTED_CLASSES,
    image: Object.freeze({
      path: "samples/boats.jpg",
      bytes: 194872,
      sha256: "8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f",
      mediaType: "image/jpeg",
      width: 1920,
      height: 1080,
      source: IMAGE_SOURCE,
    }),
    model: Object.freeze({
      path: "models/yolo26n-obb-privacy-sanitized.onnx",
      bytes: 10207127,
      sha256: "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
      mediaType: "application/onnx",
      release: "v8.4.0",
      source: MODEL_SOURCE,
      sourceSha256: "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38",
      license: "AGPL-3.0-only",
      modificationStatus: "metadata-only",
    }),
    input: Object.freeze({
      name: "images",
      type: "float32",
      dims: Object.freeze([1, 3, 1024, 1024]),
      channelOrder: "RGB",
      normalization: "divide-by-255",
      letterboxValue: 114,
    }),
    output: Object.freeze({
      name: "output0",
      type: "float32",
      dims: Object.freeze([1, "N", 7]),
      rowWidth: 7,
      layout: Object.freeze([
        "cx", "cy", "w", "h", "confidence", "class", "angleRadians",
      ]),
    }),
    provenance: Object.freeze({
      upstream: "Ultralytics YOLO26n-OBB",
      status: "privacy-sanitized AGPL derivative",
      trainingDataset: "DOTAv1",
    }),
    sanitization: Object.freeze({
      path: "third_party/yolo26n-obb-privacy-sanitization.json",
      modificationDate: "2026-08-31",
      removedMetadataEntries: 1,
      modifiedField: "ModelProto.metadata_props[0].value",
    }),
    license: Object.freeze({
      path: "third_party/ULTRALYTICS-AGPL-3.0.txt",
      bytes: 34523,
      sha256: "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
    }),
    notice: "THIRD_PARTY_NOTICES.md",
  });

  function fail(code) {
    throw new Error(code);
  }

  function exactValue(actual, expected) {
    if (Array.isArray(expected)) {
      return Array.isArray(actual) && actual.length === expected.length &&
        expected.every((item, index) => exactValue(actual[index], item));
    }
    if (expected && typeof expected === "object") {
      if (!actual || typeof actual !== "object" || Array.isArray(actual)) return false;
      const actualKeys = Object.keys(actual).sort();
      const expectedKeys = Object.keys(expected).sort();
      return exactValue(actualKeys, expectedKeys) &&
        expectedKeys.every((key) => exactValue(actual[key], expected[key]));
    }
    return typeof actual === typeof expected && Object.is(actual, expected);
  }

  function deepFreeze(value) {
    if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
    Object.values(value).forEach(deepFreeze);
    return Object.freeze(value);
  }

  function validateManifest(value) {
    let copy;
    try {
      if (!exactValue(value, EXPECTED)) fail("DEMO_MANIFEST");
      copy = JSON.parse(JSON.stringify(value));
    } catch (_error) {
      fail("DEMO_MANIFEST");
    }
    const frozen = deepFreeze(copy);
    if (!Object.isFrozen(frozen) || !Object.isFrozen(frozen.model)) {
      fail("DEMO_MANIFEST");
    }
    return frozen;
  }

  function toHex(buffer) {
    return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  async function fetchVerifiedModel(manifest, signal) {
    if (!Object.isFrozen(manifest) || !Object.isFrozen(manifest.model)) {
      fail("DEMO_MANIFEST");
    }
    const admitted = validateManifest(manifest);
    let url;
    let expectedUrl;
    try {
      url = new URL(admitted.model.path, globalThis.location.href);
      expectedUrl = new URL(EXPECTED.model.path, globalThis.location.href);
    } catch (_error) {
      fail("DEMO_MODEL_URL");
    }
    if (
      url.origin !== globalThis.location.origin ||
      url.pathname !== expectedUrl.pathname ||
      url.search ||
      url.hash
    ) {
      fail("DEMO_MODEL_URL");
    }

    let response;
    try {
      response = await fetch(url.href, {
        cache: "force-cache",
        credentials: "same-origin",
        redirect: "error",
        signal,
      });
    } catch (_error) {
      fail("DEMO_MODEL_FETCH");
    }
    if (!response.ok || !response.body) fail("DEMO_MODEL_FETCH");
    const declaredLength = response.headers.get("content-length");
    if (
      declaredLength !== null &&
      (!/^\d+$/.test(declaredLength) || Number(declaredLength) !== admitted.model.bytes)
    ) {
      fail("DEMO_MODEL_SIZE");
    }

    const reader = response.body.getReader();
    const chunks = [];
    let length = 0;
    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > MAX_MODEL_BYTES || length > admitted.model.bytes) {
          await reader.cancel();
          fail("DEMO_MODEL_SIZE");
        }
        chunks.push(value);
      }
    } catch (error) {
      if (error?.message === "DEMO_MODEL_SIZE") throw error;
      fail("DEMO_MODEL_FETCH");
    }
    if (length !== admitted.model.bytes) fail("DEMO_MODEL_SIZE");

    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    let digest;
    try {
      digest = toHex(await crypto.subtle.digest("SHA-256", bytes));
    } catch (_error) {
      fail("DEMO_MODEL_DIGEST");
    }
    if (digest !== admitted.model.sha256) fail("DEMO_MODEL_DIGEST");
    return bytes.buffer;
  }

  root.DemoAssets = Object.freeze({validateManifest, fetchVerifiedModel});
})(globalThis);
