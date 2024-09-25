from .base import BaseMemory
from ..embedding.base import BaseEmbedding
from ..storage.base import BaseStorage
from ..retrieval.semantic_search import SemanticSearch
from typing import List, Optional, Tuple
class LongTermMemory(BaseMemory):
    def __init__(self, embedding: BaseEmbedding, storage: BaseStorage, dim: int):
        self.embedding = embedding
        self.storage = storage
        self.retrieval = SemanticSearch(dim, use_faiss=True)

    def add(self, text: str):
        embedding = self.embedding.encode(text)
        self.storage.store(text, embedding)

    def get_relevant(self, query: str, k: int = 5, threshold: Optional[float] = None, 
                     return_scores: bool = False) -> List[str] | List[Tuple[str, float]]:
        query_embedding = self.embedding.encode(query)
        return self.retrieval.search(query_embedding, self.storage, k, threshold, return_scores)