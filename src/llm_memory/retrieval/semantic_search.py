from .base import BaseRetrieval
from ..storage.base import BaseStorage
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SemanticSearch(BaseRetrieval):
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss
        if use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
        self.texts = []
        self.embeddings = np.array([])

    def build_index(self, storage: BaseStorage):
        all_data = storage.get_all()
        logger.debug(f"Building index with {len(all_data)} items")
        if not all_data:
            logger.warning("No data in storage")
            self.texts = []
            self.embeddings = np.array([])
            return

        self.texts, embeddings = zip(*all_data)
        self.embeddings = np.array(embeddings).astype('float32')
        
        logger.debug(f"Embeddings shape: {self.embeddings.shape}")
        logger.debug(f"Embeddings min: {np.min(self.embeddings)}, max: {np.max(self.embeddings)}")

        if self.use_faiss:
            self.index.reset()
            self.index.add(self.embeddings)
            logger.debug("FAISS index built")

    def retrieve(self, query_embedding: np.ndarray, storage: BaseStorage, k: int,
                 threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        if self.index is None or len(self.texts) != len(storage.get_all()):
            logger.info("Rebuilding index")
            self.build_index(storage)

        if not self.texts:
            logger.warning("No texts in index")
            return []

        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        logger.debug(f"Query embedding min: {np.min(query_embedding)}, max: {np.max(query_embedding)}")

        k = min(k, len(self.texts))
        logger.debug(f"Retrieving top {k} results")

        if self.use_faiss:
            similarities, indices = self.index.search(query_embedding, k)
            logger.debug(f"FAISS similarities: {similarities}")
            results = [(self.texts[i], float(s)) for i, s in zip(indices[0], similarities[0])]
        else:
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            logger.debug(f"Cosine similarities min: {np.min(similarities)}, max: {np.max(similarities)}")
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            results = [(self.texts[i], float(similarities[i])) for i in top_k_indices]

        logger.debug(f"Results before threshold: {results}")

        if threshold is not None:
            results = [(text, score) for text, score in results if score >= threshold]
            logger.debug(f"Results after threshold {threshold}: {results}")

        return results

    def search(self, query_embedding: np.ndarray, storage: BaseStorage,
               k: int = 5, threshold: Optional[float] = None,
               return_scores: bool = False) -> List[str] | List[Tuple[str, float]]:
        results = self.retrieve(query_embedding, storage, k, threshold)
        if return_scores:
            return results
        else:
            return [text for text, _ in results]