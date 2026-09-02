# Third-party notices

## Public-domain NAIP aerial derivatives

This public-domain USGS/USDA NAIP harbor derivative is a curated integration example, not ground
truth, accuracy, evaluation, benchmark, representative-dataset, model-quality, or latency evidence.
It is checked only for result drift at the shared 0.25 threshold; no per-image tuning or precomputed
result is used. Crop/resample and metadata removal produced the final sRGB JPEG. No USGS or USDA endorsement is implied.

- **低密度港區航拍範例** — `samples/harbor.jpg`; USGS-NAIP product `m_3411955_sw_11_060_20220514`, 2022, acquired 2022-05-14, USDA, [service](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer), [Public Domain record](https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39); Public Domain; bbox `[-119.216719, 34.14417, -119.200719, 34.15417]`; crop/resample/metadata removal; 241046 bytes, SHA-256 `916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0`.

## Model license

The bundled model is a privacy-sanitized AGPL derivative of Ultralytics YOLO26n-OBB release v8.4.0.
One non-inference metadata entry was removed on 2026-08-31; graph and weights were verified unchanged.
The model was trained on DOTAv1. The complete AGPL-3.0-only text is
`third_party/ULTRALYTICS-AGPL-3.0.txt`; the transformation record is
`third_party/yolo26n-obb-privacy-sanitization.json`. No Ultralytics endorsement or commercial-use
clearance is implied. License SHA-256: `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
