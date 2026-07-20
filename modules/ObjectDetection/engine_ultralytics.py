import functools, numpy as np, torch
# Trusted local model files (Ultralytics/community .pt): restore torch<2.6 default.
_orig = torch.load
torch.load = functools.partial(_orig, weights_only=False)
from ultralytics import YOLO

class UltralyticsEngine:
    def __init__(self, model_path_or_id, use_cuda):
        self.device = 0 if use_cuda else "cpu"
        self.model = YOLO(model_path_or_id)   # auto-downloads known ids; loads local .pt

    def detect(self, image, min_confidence):
        r = self.model.predict(source=image, conf=float(min_confidence),
                               device=self.device, verbose=False)[0]
        names = r.names
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append({
                "confidence": float(b.conf[0]),
                "label": str(names[int(b.cls[0])]),
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
            })
        return out
