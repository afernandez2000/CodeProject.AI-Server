import importlib.util, os, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("model_router", os.path.join(BASE,"model_router.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
LEGACY = os.path.join(BASE, "custom-models", "ipcam-combined.pt")

@pytest.mark.skipif(not os.path.exists(LEGACY), reason="legacy model not downloaded")
def test_legacy_routes_to_yolov5():
    R = _load().ModelRouter(use_cuda=False)
    eng = R.get(LEGACY)
    assert type(eng).__name__ == "Yolov5Engine"
    # cached: same instance on second call
    assert R.get(LEGACY) is eng

def test_modern_routes_to_ultralytics(tmp_path):
    # yolo26n.pt is a modern Ultralytics weight
    R = _load().ModelRouter(use_cuda=False)
    eng = R.get("yolo26n.pt")
    assert type(eng).__name__ == "UltralyticsEngine"
