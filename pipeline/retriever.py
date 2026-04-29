import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    """Load once, reuse. Downloads ~80MB on first run."""
    global _embedder
    if _embedder is None:
        print("Loading sentence embedder (first time ~30s)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedder ready.")
    return _embedder


class BM25Retriever:
    """
    Keyword-based retrieval using BM25Okapi.
    Fast, no model needed. Sensitive to vocabulary drift — 
    when query words change across hops, recall drops.
    """
    def __init__(self, corpus):
        self.corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=5):
        scores = self.bm25.get_scores(query.lower().split())
        top_k = min(top_k, len(self.corpus))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.corpus[i] for i in top_indices]

    def encode(self, text):
        """Encode text to vector using shared embedder."""
        return get_embedder().encode([text])[0]


class FAISSRetriever:
    """
    Neural semantic retrieval using sentence embeddings + FAISS index.
    Handles vocabulary drift better than BM25 because it matches
    meaning rather than exact words.
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self.embedder = get_embedder()
        vecs = self.embedder.encode(
            corpus, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def retrieve(self, query, top_k=5):
        top_k = min(top_k, len(self.corpus))
        q_vec = self.embedder.encode(
            [query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, indices = self.index.search(q_vec, top_k)
        return [self.corpus[i] for i in indices[0] if i >= 0]

    def encode(self, text):
        return self.embedder.encode([text])[0]