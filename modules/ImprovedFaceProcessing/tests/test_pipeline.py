import importlib.util, os, cv2, pytest
BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
REC  = os.path.join(BASE, "assets", "adaface_ir101.pt")

def _mod():
    spec = importlib.util.spec_from_file_location("improved_face", os.path.join(BASE,"intelligencelayer","improved_face.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.mark.skipif(not (os.path.exists(DET) and os.path.exists(REC)), reason="weights not downloaded")
def test_register_then_recognize(tmp_path):
    # Similarity is on remapped (cos+1)/2 scale; same-person Hemsworth pair
    # yields remapped sim well above 0.64; cross-person (Scarlett) falls below.
    P = _mod().Pipeline(DET, REC, "ir_101", use_cuda=False, db_path=str(tmp_path/"g.db"), threshold=0.64)
    ref = cv2.imread("src/demos/TestData/Faces/Chris-Hemsworth-2.jpg")
    assert P.register(ref, "chris")["success"] is True
    out = P.recognize(cv2.imread("src/demos/TestData/Faces/chris-hemsworth-1.jpg"), threshold=0.64)
    assert out["success"] is True
    ids = [p["userid"] for p in out["predictions"]]
    assert "chris" in ids

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_detect_shape():
    P = _mod().Pipeline(DET, REC, "ir_101", use_cuda=False, db_path=":memory:", threshold=0.64)
    out = P.detect(cv2.imread("src/demos/TestData/Faces/family-on-couch.jpg"))
    assert out["success"] is True and out["count"] >= 3
    assert {"confidence","x_min","y_min","x_max","y_max"} <= set(out["predictions"][0])

@pytest.mark.skipif(not os.path.exists(DET), reason="weights not downloaded")
def test_detect_threshold_filters():
    """High min_confidence must return <= faces than low min_confidence."""
    P = _mod().Pipeline(DET, REC, "ir_101", use_cuda=False, db_path=":memory:", threshold=0.64)
    img = cv2.imread("src/demos/TestData/Faces/family-on-couch.jpg")
    out_low  = P.detect(img, threshold=0.4)
    out_high = P.detect(img, threshold=0.99)
    print(f"\nthreshold=0.4  → {out_low['count']} faces")
    print(f"threshold=0.99 → {out_high['count']} faces")
    assert out_high["count"] <= out_low["count"], (
        f"Expected fewer faces at thr=0.99 ({out_high['count']}) than "
        f"thr=0.4 ({out_low['count']})"
    )
