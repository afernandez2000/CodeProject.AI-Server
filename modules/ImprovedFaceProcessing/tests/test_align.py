import importlib.util, os, cv2, numpy as np, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
IMG  = "src/demos/TestData/Faces/family-on-couch.jpg"

def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE,"intelligencelayer",name+".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_align_returns_112():
    det = _mod("detector").ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    face = det.detect(cv2.imread(IMG))[0]
    crop = _mod("align").align_face(cv2.imread(IMG), face.kps)
    assert crop.shape == (112, 112, 3)
