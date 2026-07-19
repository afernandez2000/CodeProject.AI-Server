# modules/ImprovedFaceProcessing/tests/test_modulesettings.py
import json, re, os
BASE = os.path.dirname(os.path.dirname(__file__))

def _load_settings():
    raw = open(os.path.join(BASE, "modulesettings.json"), encoding="utf-8").read()
    raw = re.sub(r"(?<!:)//.*", "", raw)          # strip // comments (preserve https://)
    return json.loads(raw)

def test_module_identity_and_routes():
    d = _load_settings()
    m = d["Modules"]["ImprovedFaceProcessing"]
    assert m["Version"] == "1.0.0"
    assert m["LaunchSettings"]["Queue"] == "improvedfaceprocessing_queue"
    assert m["LaunchSettings"]["Runtime"] == "python3.11"
    routes = {r["Route"] for r in m["RouteMaps"]}
    assert routes == {"vision/face", "vision/face/recognize", "vision/face/register",
                      "vision/face/match", "vision/face/list", "vision/face/delete"}

def test_download_tiers_defined():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "download_models", os.path.join(BASE, "intelligencelayer", "download_models.py"))
    dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
    for tier in ("accurate", "fast"):
        t = dm.TIER_MODELS[tier]
        assert t["detector"]["file"].endswith(".onnx")
        assert t["recognizer"]["file"].endswith(".pt")
        assert t["detector"]["url"].startswith("http")
        assert t["recognizer"]["url"].startswith("http")
