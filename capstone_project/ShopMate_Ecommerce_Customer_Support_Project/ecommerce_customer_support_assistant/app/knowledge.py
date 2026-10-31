
"""Small local RAG store: sentence-transformer embeddings when installed,
with a TF-IDF fallback so the demo can still run in a lightweight environment."""
import json
from pathlib import Path
import numpy as np

class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = path
        self.docs = json.loads(path.read_text(encoding="utf-8"))
        self.texts = [f"{d['title']}. {d['content']}" for d in self.docs]
        self.mode = "tfidf"
        self.model = None
        self.matrix = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.matrix = self.model.encode(self.texts, normalize_embeddings=True)
            self.mode = "sentence-transformers"
        except Exception:
            # Offline / first-run fallback: lexical vector retrieval.
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(self.texts)
    def search(self, query: str, k: int = 3):
        if self.mode == "sentence-transformers":
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = np.asarray(self.matrix) @ q
            order = np.argsort(scores)[::-1][:k]
            return [{"title": self.docs[i]["title"], "content": self.docs[i]["content"],
                     "score": float(scores[i])} for i in order]
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(scores)[::-1][:k]
        return [{"title": self.docs[i]["title"], "content": self.docs[i]["content"],
                 "score": float(scores[i])} for i in order]
