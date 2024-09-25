from .base import BaseRetrieval
from ..storage.base import BaseStorage
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class SemanticSearch(BaseRetrieval):
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss
        if use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def build_index(self, storage: BaseStorage):
        all_data = storage.get_all()
        texts, embeddings = zip(*all_data)
        if self.use_faiss:
            self.index.reset()
            self.index.add(np.array(embeddings).astype('float32'))
        self.texts = texts
        self.embeddings = np.array(embeddings)

    def retrieve(self, query_embedding: np.ndarray, storage: BaseStorage, k: int, 
                 threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        if self.index is None or len(self.texts) != storage.count():
            self.build_index(storage)

        if self.use_faiss:
            similarities, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
            results = [(self.texts[i], float(s)) for i, s in zip(indices[0], similarities[0])]
        else:
            similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            results = [(self.texts[i], float(similarities[i])) for i in top_k_indices]

        if threshold is not None:
            results = [(text, score) for text, score in results if score >= threshold]

        return results

    def search(self, query_embedding: np.ndarray, storage: BaseStorage, 
               k: int = 5, threshold: Optional[float] = None, 
               return_scores: bool = False) -> List[str] | List[Tuple[str, float]]:
        results = self.retrieve(query_embedding, storage, k, threshold)
        if return_scores:
            return results
        else:
            return [text for text, _ in results]
