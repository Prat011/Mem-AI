from typing import List, Dict, Any, Optional
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .embedding.base import BaseEmbedding
from .storage.base import BaseStorage
from .retrieval.base import BaseRetrieval

class LLMMemoryManager:
    def __init__(self, embedding: BaseEmbedding, storage: BaseStorage, retrieval: BaseRetrieval):
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory(embedding, storage, retrieval)

    def add_to_memory(self, text: str, is_user_input: bool = False):
        self.short_term_memory.add(text)
        if is_user_input:
            self.long_term_memory.add(text)

    def get_relevant_memories(self, query: str, k: int = 5) -> List[str]:
        stm_results = self.short_term_memory.get_relevant(query)
        ltm_results = self.long_term_memory.get_relevant(query, k)
        return stm_results + ltm_results

    def get_context(self, query: str) -> str:
        relevant_memories = self.get_relevant_memories(query)
        return "\n".join(relevant_memories)

    def clear_short_term_memory(self):
        self.short_term_memory.clear()

    def get_relevant_memories(self, query: str, k: int = 5, threshold: Optional[float] = None, 
                              return_scores: bool = False) -> List[str]:
        stm_results = self.short_term_memory.get_relevant(query)
        ltm_results = self.long_term_memory.get_relevant(query, k, threshold, return_scores)
        
        if return_scores:
            # Assign perfect scores to short-term memory results
            stm_results_with_scores = [(text, 1.0) for text in stm_results]
            return stm_results_with_scores + ltm_results
        else:
            return stm_results + ltm_results

    def get_context(self, query: str, k: int = 5, threshold: Optional[float] = None) -> str:
        relevant_memories = self.get_relevant_memories(query, k, threshold)
        return "\n".join(relevant_memories)