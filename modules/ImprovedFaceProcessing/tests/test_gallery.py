import importlib.util, os, numpy as np, tempfile
BASE = os.path.dirname(os.path.dirname(__file__))
def _load():
    spec = importlib.util.spec_from_file_location("gallery", os.path.join(BASE,"intelligencelayer","gallery.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_add_list_match_delete():
    G = _load().Gallery(os.path.join(tempfile.mkdtemp(), "t.db"))
    a = np.ones(512, dtype="float32");  a /= np.linalg.norm(a)
    b = np.arange(512, dtype="float32"); b /= np.linalg.norm(b)
    G.add("alice", a); G.add("bob", b); G.load()
    assert set(G.list_ids()) == {"alice", "bob"}
    # Similarity is now on remapped (cos+1)/2 scale [0,1].
    # Identical unit vectors → raw_cos=1.0 → remapped=1.0; threshold 0.64 should match.
    uid, sim = G.match(a, threshold=0.64)
    assert uid == "alice" and sim > 0.99
    # Zero vector → raw_cos=0.0 → remapped=0.5, which is below threshold 0.64 → "unknown".
    uid2, _ = G.match(np.zeros(512, dtype="float32"), threshold=0.64)
    assert uid2 == "unknown"
    assert G.delete("alice") == 1
    G.load(); assert G.list_ids() == ["bob"]
