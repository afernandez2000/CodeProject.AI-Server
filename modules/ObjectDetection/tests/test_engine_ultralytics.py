import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("engine_ultralytics", os.path.join(BASE,"engine_ultralytics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
IMG = "src/demos/TestData/Objects/street-at-night.jpg"

def test_detect_returns_predictions():
    E = _load().UltralyticsEngine("yolo26n.pt", use_cuda=False)
    preds = E.detect(cv2.imread(IMG), 0.4)
    assert isinstance(preds, list) and len(preds) >= 1
    p = preds[0]
    assert {"confidence","label","x_min","y_min","x_max","y_max"} <= set(p)
    assert isinstance(p["label"], str) and 0.0 <= p["confidence"] <= 1.0
