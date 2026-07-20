import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.path.join(BASE, "custom-models", "ipcam-combined.pt")
def _load():
    spec = importlib.util.spec_from_file_location("engine_yolov5", os.path.join(BASE,"engine_yolov5.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(MODEL), reason="legacy IPcam model not downloaded")
def test_legacy_detect():
    E = _load().Yolov5Engine(MODEL, use_cuda=False)
    preds = E.detect(cv2.imread("src/demos/TestData/Objects/study-group.jpg"), 0.4)
    assert isinstance(preds, list)
    if preds:
        assert {"confidence","label","x_min","y_min","x_max","y_max"} <= set(preds[0])
