from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VectorStore:
    def __init__(self, embeddings, metadata):
        self.vectors = embeddings
        self.meta = metadata
    def query(self, vector, top_k=10):
        sims = cosine_similarity([vector], self.vectors)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [(self.meta[i], float(sims[i])) for i in idxs]
