# modules/ObjectDetection/tests/test_modulesettings.py
import json, re, os
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    raw = re.sub(r"(?<!:)//.*", "", open(os.path.join(BASE,"modulesettings.json"),encoding="utf-8").read())
    return json.loads(raw)
def test_identity_routes():
    m = _load()["Modules"]["ObjectDetection"]
    assert m["Version"] == "1.0.0"
    assert m["LaunchSettings"]["Queue"] == "objectdetection_queue"
    assert m["LaunchSettings"]["Runtime"] == "python3.11"
    routes = {r["Route"] for r in m["RouteMaps"]}
    assert routes == {"vision/detection", "vision/custom", "vision/custom/list"}
    cmds = {r["Command"] for r in m["RouteMaps"]}
    assert cmds == {"detect", "custom", "list-custom"}
    ev = m["EnvironmentVariables"]
    assert "DEFAULT_MODEL" in ev and "MODEL_TIER" in ev
