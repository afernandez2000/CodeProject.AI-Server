import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
IMG  = "src/demos/TestData/Faces/family-on-couch.jpg"

def _load():
    spec = importlib.util.spec_from_file_location("detector", os.path.join(BASE,"intelligencelayer","detector.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_detect_finds_faces_with_landmarks():
    det = _load().ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    faces = det.detect(cv2.imread(IMG))
    assert len(faces) >= 3
    f = faces[0]
    assert f.kps.shape == (5, 2)
    assert len(f.bbox) == 4 and f.score > 0.3
