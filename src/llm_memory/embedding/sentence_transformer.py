from .base import BaseEmbedding
import numpy as np
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)