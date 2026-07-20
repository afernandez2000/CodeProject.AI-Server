import importlib.util, os, cv2, pytest

BASE = os.path.dirname(os.path.dirname(__file__))

def _mod():
    spec = importlib.util.spec_from_file_location("detect_adapter", os.path.join(BASE, "detect_adapter.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

class _Opts:  # minimal stand-in
    use_CUDA = False
    tier = "fast"
    default_model = "yolo26"
    default_engine = "ultralytics"
    default_weight = "yolo26n.pt"
    models_dir = os.path.join(BASE, "assets")
    custom_models_dir = os.path.join(BASE, "custom-models")
    class_filter = []
    min_confidence_default = 0.4


def test_detect_shape():
    D = _mod().Detector(_Opts())
    out = D.detect(cv2.imread("src/demos/TestData/Objects/street-at-night.jpg"), 0.4)
    assert out["success"] is True
    assert {"count", "predictions", "inferenceMs", "processMs"} <= set(out)
    if out["predictions"]:
        assert {"confidence", "label", "x_min", "y_min", "x_max", "y_max"} <= set(out["predictions"][0])


def test_class_filter():
    o = _Opts()
    o.class_filter = ["car"]
    D = _mod().Detector(o)
    out = D.detect(cv2.imread("src/demos/TestData/Objects/traffic.jpg"), 0.3)
    assert all(p["label"].lower() == "car" for p in out["predictions"])


def test_list_models_shape():
    D = _mod().Detector(_Opts())
    out = D.list_models()
    assert set(out.keys()) == {"success", "models"}, (
        f"list_models() returned unexpected keys: {set(out.keys())}"
    )
    assert out["success"] is True
    assert isinstance(out["models"], list)
