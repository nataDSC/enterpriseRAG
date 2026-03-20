import numpy as np

from enterprise_rag.text_utils import tokenize


class HashingEmbedder:
    """Fast, deterministic embedding for demo use (no model downloads)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            idx = hash(token) % self.dimension
            sign = 1.0 if (hash(token + "_sign") % 2 == 0) else -1.0
            vector[idx] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector
