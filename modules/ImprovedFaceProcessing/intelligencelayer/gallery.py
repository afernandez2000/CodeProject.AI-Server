import os, sqlite3, threading, numpy as np, ast

CREATE = "CREATE TABLE IF NOT EXISTS TB_EMBEDDINGS(userid TEXT PRIMARY KEY, embedding TEXT NOT NULL)"

class Gallery:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ids = []
        self._mat = None                      # (N,512) float32
        with sqlite3.connect(self.db_path) as c:
            c.execute(CREATE)

    def add(self, userid, embedding):
        text = repr(embedding.astype("float32").tolist())
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO TB_EMBEDDINGS(userid,embedding) VALUES(?,?) "
                      "ON CONFLICT(userid) DO UPDATE SET embedding=excluded.embedding", (userid, text))

    def delete(self, userid):
        with sqlite3.connect(self.db_path) as c:
            cur = c.execute("DELETE FROM TB_EMBEDDINGS WHERE userid=?", (userid,))
            return cur.rowcount

    def list_ids(self):
        with sqlite3.connect(self.db_path) as c:
            return [r[0] for r in c.execute("SELECT userid FROM TB_EMBEDDINGS")]

    def load(self):
        ids, vecs = [], []
        with sqlite3.connect(self.db_path) as c:
            for uid, text in c.execute("SELECT userid, embedding FROM TB_EMBEDDINGS"):
                ids.append(uid); vecs.append(np.asarray(ast.literal_eval(text), dtype="float32"))
        with self._lock:
            self._ids = ids
            self._mat = np.stack(vecs) if vecs else None

    def match(self, embedding, threshold):
        with self._lock:
            ids, mat = self._ids, self._mat
        if mat is None or len(ids) == 0:
            return "unknown", 0.0
        sims = mat @ embedding.astype("float32")
        i = int(sims.argmax()); best = float(sims[i])
        return (ids[i], best) if best >= threshold else ("unknown", best)
