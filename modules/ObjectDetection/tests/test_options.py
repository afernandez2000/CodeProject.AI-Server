import importlib.util, os
BASE = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("options", os.path.join(BASE,"options.py"))
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)

def test_tiers():
    assert o.select_tier(True, 32*10**9, "auto") == "accurate"
    assert o.select_tier(True,  6*10**9, "auto") == "balanced"
    assert o.select_tier(False, 0,       "auto") == "fast"
    assert o.select_tier(True,  6*10**9, "accurate") == "accurate"

def test_resolve_default_cctv():
    # CCTV family uses yolov5 engine on GPU tiers
    eng, w = o.resolve_default("ipcam-combined", "accurate")
    assert eng == "yolov5" and "ipcam" in w
    # CPU/fast tier for CCTV falls back to the generic nano (ultralytics)
    eng2, w2 = o.resolve_default("ipcam-combined", "fast")
    assert eng2 == "ultralytics"

def test_resolve_default_generic():
    eng, w = o.resolve_default("yolo26", "accurate")
    assert eng == "ultralytics"
