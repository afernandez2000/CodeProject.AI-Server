import functools, numpy as np, torch
_orig = torch.load
torch.load = functools.partial(_orig, weights_only=False)   # legacy trusted weights (torch 2.13)
from yolov5.models.common import DetectMultiBackend, AutoShape

class Yolov5Engine:
    def __init__(self, model_path, use_cuda):
        dev = torch.device("cuda:0" if use_cuda else "cpu")
        self.model = AutoShape(DetectMultiBackend(model_path, device=dev, fp16=False))

    def detect(self, bgr, min_confidence):
        self.model.conf = float(min_confidence)
        r = self.model(bgr[:, :, ::-1], size=640)   # AutoShape expects RGB
        names = self.model.names
        out = []
        for *xyxy, conf, cls in r.xyxy[0].tolist():
            out.append({
                "confidence": float(conf),
                "label": str(names[int(cls)]),
                "x_min": int(xyxy[0]), "y_min": int(xyxy[1]),
                "x_max": int(xyxy[2]), "y_max": int(xyxy[3]),
            })
        return out
