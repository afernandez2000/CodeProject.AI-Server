# Task 1 Spike Findings

Date: 2026-07-19

## Spike 1 — Engine coexistence + YOLO26 availability

**Command run:**
```
runtimes/bin/ubuntu/python311/venv/bin/python -m pip install -q ultralytics
# then import test + model probe
```

**Results:**
- `ultralytics 8.4.102` installed successfully into the shared venv.
- `torch 2.13.0+cu130`, `ultralytics 8.4.102`, and `yolov5` (6.2.3) all import together with no conflicts.
- `yolo26n.pt` was tried first and succeeded — Ultralytics downloads it from
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` (5.3 MB).

**Resolved generic model id: `yolo26n.pt`** (YOLO26 is available in ultralytics 8.4.102 — no fallback needed).

## Spike 2 — Legacy IPcam model load under yolov5==6.2.3

**Model tested:** `modules/ObjectDetectionYOLOv5-6.2/custom-models/ipcam-combined.pt`  
(copied to `modules/ObjectDetection/custom-models/ipcam-combined.pt` for Task 4/5 use)

**Load method:**
```python
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)  # legacy trusted weights
from yolov5.models.common import DetectMultiBackend
m = DetectMultiBackend("<path>.pt", device=torch.device("cpu"))
```

**Result:** Load succeeded — `YOLOv5s summary: 283 layers, 7314428 parameters`

**Class names (first 12):**
`person, bicycle, car, motorcycle, bus, truck, bird, cat, dog, horse, sheep, cow`
(80-class COCO-derived subset tuned for CCTV/IPcam scenes)

## Spike 3 — IPcam custom-model download source

**install.sh line:**
```bash
getFromServer "models/" "custom-models-yolo5-pt.zip" "custom-models" "Downloading Custom YOLO models..."
```

**AssetStorageUrl** (from `src/server/appsettings.json`):
```
https://codeproject-ai-bunny.b-cdn.net/server/assets/
```

**Full download URL:**
```
https://codeproject-ai-bunny.b-cdn.net/server/assets/models/custom-models-yolo5-pt.zip
```

**IPcam model filenames present in the zip** (confirmed on disk in ObjectDetectionYOLOv5-6.2/custom-models/):
- `ipcam-combined.pt`   — 80-class CCTV combined model (person, vehicle, animals, etc.)
- `ipcam-general.pt`    — general IPcam variant
- `ipcam-animal.pt`     — animal-focused variant
- `ipcam-dark.pt`       — low-light / dark scene variant
- `license-plate.pt`    — license plate detector
- `actionnetv2.pt`      — action recognition model

**Size variants:** No nano/small/large size variants observed for IPcam models — each is a single `.pt` file (fixed architecture, YOLOv5s-based ~7.3M params for ipcam-combined). The standard YOLO models (yolov5n/s/m/l/x) have size variants but are in a separate zip (`models-yolo5-pt.zip`).

**Note for Task 7:** The zip asset to automate is `custom-models-yolo5-pt.zip` from the CDN URL above. No `delivery.pt` variant was found in the current downloaded set.
