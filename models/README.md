# Hyakkiyakou Patch Model

This directory is reserved for a local detector that recognizes shikigami missing
from the bundled `oashya` model.

Place files here:

- `hya_patch_fp32.onnx`: an ONNX detector exported in YOLO-style output.
- `hya_patch_labels.json`: labels in the same order as the detector classes.

Example label file:

```json
[
  {"id": 219, "label": "sp_036", "name": "新SP式神", "rarity": "sp"},
  {"id": 220, "label": "ssr_047", "name": "新SSR式神", "rarity": "ssr"}
]
```

When both files exist, `tasks.Hyakkiyakou.detector.Tracker` runs the bundled
`oashya` detector first and then merges this patch detector's results.
