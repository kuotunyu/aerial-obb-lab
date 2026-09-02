(function exposeDemoAssets(root) {
  "use strict";
  const MAX_MODEL_BYTES = 15 * 1024 * 1024;
  const SAMPLE_IDS = Object.freeze(["harbor"]);
  function fail(code) { throw new Error(code); }
  function exactValue(actual, expected) {
    if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((item, index) => exactValue(actual[index], item));
    if (expected && typeof expected === "object") {
      if (!actual || typeof actual !== "object" || Array.isArray(actual)) return false;
      const actualKeys = Object.keys(actual).sort(); const expectedKeys = Object.keys(expected).sort();
      return exactValue(actualKeys, expectedKeys) && expectedKeys.every((key) => exactValue(actual[key], expected[key]));
    }
    return typeof actual === typeof expected && Object.is(actual, expected);
  }
  function deepFreeze(value) { if (!value || typeof value !== "object" || Object.isFrozen(value)) return value; Object.values(value).forEach(deepFreeze); return Object.freeze(value); }
  const EXPECTED = deepFreeze({"classes":["plane","ship","storage tank","baseball diamond","tennis court","basketball court","ground track field","harbor","bridge","large vehicle","small vehicle","helicopter","roundabout","soccer ball field","swimming pool"],"defaultConfidence":0.25,"id":"ultralytics-yolo26n-obb-demo","input":{"channelOrder":"RGB","dims":[1,3,1024,1024],"letterboxValue":114,"name":"images","normalization":"divide-by-255","type":"float32"},"license":{"bytes":34523,"path":"third_party/ULTRALYTICS-AGPL-3.0.txt","sha256":"0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"},"model":{"bytes":10207127,"license":"AGPL-3.0-only","mediaType":"application/onnx","modificationStatus":"metadata-only","path":"models/yolo26n-obb-privacy-sanitized.onnx","release":"v8.4.0","sha256":"a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97","source":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx","sourceSha256":"02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"},"notice":"THIRD_PARTY_NOTICES.md","output":{"dims":[1,"N",7],"layout":["cx","cy","w","h","confidence","class","angleRadians"],"name":"output0","rowWidth":7,"type":"float32"},"provenance":{"status":"privacy-sanitized AGPL derivative","trainingDataset":"DOTAv1","upstream":"Ultralytics YOLO26n-OBB"},"samples":[{"alt":"低密度港區的真實航拍原圖","bytes":241046,"derivation":{"bboxWgs84":[-119.216719,34.14417,-119.200719,34.15417],"color":"sRGB","jpegQuality":90,"metadata":"stripped","outputSize":[1280,800]},"guardrails":{"classIds":[1,2,7],"countMax":26,"countMin":16,"representative":{"classId":1,"cx":951.1,"cy":443.1,"h":26.4,"tolerance":32.9,"w":164.4}},"height":800,"id":"harbor","mediaType":"image/jpeg","path":"samples/harbor.jpg","sha256":"916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0","source":{"acquisitionDate":1652486400000,"agency":"USDA","productId":"m_3411955_sw_11_060_20220514","publicDomainRecord":"https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39","service":"https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer","year":2022},"title":"低密度港區航拍範例","width":1280}],"sanitization":{"modificationDate":"2026-08-31","modifiedField":"ModelProto.metadata_props[0].value","path":"third_party/yolo26n-obb-privacy-sanitization.json","removedMetadataEntries":1},"schemaVersion":2});
  function validateManifest(value) {
    let copy; try { if (!exactValue(value, EXPECTED)) fail("DEMO_MANIFEST"); copy = JSON.parse(JSON.stringify(value)); } catch (_error) { fail("DEMO_MANIFEST"); }
    return deepFreeze(copy);
  }
  function getDemoSample() { return EXPECTED.samples[0]; }
  function toHex(buffer) { return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join(""); }
  async function fetchVerifiedModel(manifest, options) {
    if (!Object.isFrozen(manifest) || !Object.isFrozen(manifest.model)) fail("DEMO_MANIFEST");
    const admitted = validateManifest(manifest); let url; let expectedUrl;
    try { url = new URL(admitted.model.path, globalThis.location.href); expectedUrl = new URL(EXPECTED.model.path, globalThis.location.href); } catch (_error) { fail("DEMO_MODEL_URL"); }
    if (url.origin !== globalThis.location.origin || url.pathname !== expectedUrl.pathname || url.search || url.hash) fail("DEMO_MODEL_URL");
    let response; try { response = await fetch(url.href, {cache:"force-cache", credentials:"same-origin", redirect:"error", signal: options?.signal ?? options}); } catch (_error) { fail("DEMO_MODEL_FETCH"); }
    if (!response.ok || !response.body) fail("DEMO_MODEL_FETCH"); const declaredLength = response.headers.get("content-length");
    if (declaredLength !== null && (!/^\d+$/.test(declaredLength) || Number(declaredLength) !== admitted.model.bytes)) fail("DEMO_MODEL_SIZE");
    const reader = response.body.getReader(); const chunks = []; let length = 0;
    try { while (true) { const {done, value} = await reader.read(); if (done) break; length += value.byteLength; if (length > MAX_MODEL_BYTES || length > admitted.model.bytes) { await reader.cancel(); fail("DEMO_MODEL_SIZE"); } chunks.push(value); } } catch (error) { if (error?.message === "DEMO_MODEL_SIZE") throw error; fail("DEMO_MODEL_FETCH"); }
    if (length !== admitted.model.bytes) fail("DEMO_MODEL_SIZE"); const bytes = new Uint8Array(length); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    let digest; try { digest = toHex(await crypto.subtle.digest("SHA-256", bytes)); } catch (_error) { fail("DEMO_MODEL_DIGEST"); } if (digest !== admitted.model.sha256) fail("DEMO_MODEL_DIGEST"); return bytes.buffer;
  }
  root.DemoAssets = Object.freeze({validateManifest, fetchVerifiedModel, getDemoSample});
})(globalThis);
