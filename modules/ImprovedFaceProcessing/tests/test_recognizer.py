import importlib.util, os, cv2, numpy as np, pytest

BASE = os.path.dirname(os.path.dirname(__file__))
DET  = os.path.join(BASE, "assets", "scrfd_10g.onnx")
REC  = os.path.join(BASE, "assets", "adaface_ir101.pt")

# Root of the repository (three levels up from the module dir)
REPO_ROOT = os.path.normpath(os.path.join(BASE, "..", ".."))


def _mod(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(BASE, "intelligencelayer", name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(
    not (os.path.exists(DET) and os.path.exists(REC)),
    reason="weights not downloaded",
)
def test_embedding_is_normalized_and_discriminative():
    det   = _mod("detector").ScrfdDetector(DET, ["CPUExecutionProvider"], 640)
    align = _mod("align").align_face
    rec   = _mod("recognizer").AdaFaceRecognizer(REC, "ir_101", use_cuda=False)

    def emb(rel_path):
        path = os.path.join(REPO_ROOT, rel_path)
        img  = cv2.imread(path)
        assert img is not None, f"Could not read image: {path}"
        faces = det.detect(img)
        assert len(faces) > 0, f"No face detected in {path}"
        f = faces[0]
        return rec.embed(align(img, f.kps))

    e1 = emb("src/demos/TestData/Faces/Chris-Hemsworth-2.jpg")
    e2 = emb("src/demos/TestData/Faces/chris-hemsworth-1.jpg")
    e3 = emb("src/demos/TestData/Faces/scarlett-johanson-1.jpg")

    assert e1.shape == (512,), f"Expected (512,), got {e1.shape}"
    assert abs(np.linalg.norm(e1) - 1.0) < 1e-3, f"Not L2-normalised: norm={np.linalg.norm(e1)}"

    same = float(e1 @ e2)
    diff = float(e1 @ e3)
    print(f"\n  same-person similarity (Hemsworth): {same:.4f}")
    print(f"  diff-person similarity (Hemsworth vs Scarlett): {diff:.4f}")

    assert same > diff and same > 0.3, (
        f"Discriminative test failed: same={same:.4f}, diff={diff:.4f}"
    )
