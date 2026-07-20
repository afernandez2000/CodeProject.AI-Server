import os, sys
# Ensure onnxruntime can find the venv's cu13 CUDA libs BEFORE it is imported
# (cross-platform: Linux LD_LIBRARY_PATH re-exec / Windows add_dll_directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cuda_libpath import ensure_cuda_libpath
ensure_cuda_libpath()

from collections import namedtuple
from insightface.model_zoo import model_zoo

Face = namedtuple("Face", ["bbox", "score", "kps"])

class ScrfdDetector:
    def __init__(self, model_path, providers, det_size=640, provider_options=None):
        kwargs = {"providers": providers}
        # provider_options (e.g. gpu_mem_limit) flow through insightface's
        # model_zoo -> PickableInferenceSession -> onnxruntime.InferenceSession.
        # Only pass it when set so the default path stays byte-identical.
        if provider_options is not None:
            kwargs["provider_options"] = provider_options
        self.model = model_zoo.get_model(model_path, **kwargs)
        ctx_id = 0 if any("CUDA" in p for p in providers) else -1
        self.model.prepare(ctx_id=ctx_id, input_size=(det_size, det_size))

    def detect(self, bgr_img):
        bboxes, kpss = self.model.detect(bgr_img, max_num=0, metric="default")
        faces = []
        for i, b in enumerate(bboxes):
            x1, y1, x2, y2, score = b
            kps = kpss[i] if kpss is not None else None
            faces.append(Face((int(x1), int(y1), int(x2), int(y2)), float(score), kps))
        return faces
