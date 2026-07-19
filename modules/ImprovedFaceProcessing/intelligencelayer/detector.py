import os
# Ensure onnxruntime can find torch's cu13 CUDA libs BEFORE it is imported.
def _add_cu13_libpath():
    here = os.path.dirname(os.path.realpath(__file__))
    venv = os.path.normpath(os.path.join(here, "..", "..", "..",
             "runtimes", "bin", "ubuntu", "python311", "venv",
             "lib", "python3.11", "site-packages", "nvidia", "cu13", "lib"))
    if os.path.isdir(venv):
        os.environ["LD_LIBRARY_PATH"] = venv + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
_add_cu13_libpath()

from collections import namedtuple
from insightface.model_zoo import model_zoo

Face = namedtuple("Face", ["bbox", "score", "kps"])

class ScrfdDetector:
    def __init__(self, model_path, providers, det_size=640):
        self.model = model_zoo.get_model(model_path, providers=providers)
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
