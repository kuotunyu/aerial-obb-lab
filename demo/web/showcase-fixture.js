(function expose(root, factory) {
  const fixture = factory();
  if (typeof module === "object" && module.exports) module.exports = fixture;
  root.OBB_SHOWCASE = fixture;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildFixture() {
  "use strict";
  return Object.freeze({
    schemaVersion: 1,
    provenance: "Committed synthetic fixture",
    imageUrl: "fixtures/showcase.svg",
    imageWidth: 400,
    imageHeight: 200,
    targetSize: 1024,
    results: Object.freeze({
      output0: Object.freeze({
        dims: Object.freeze([1, 2, 7]),
        data: Float32Array.from([
          512, 512, 256, 128, 0.9, 1, Math.PI / 2,
          100, 100, 50, 40, 0.2, 2, 0,
        ]),
      }),
    }),
  });
});
