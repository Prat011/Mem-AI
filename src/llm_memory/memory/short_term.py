from .base import BaseMemory
from ..utils.text_processing import simple_similarity

class ShortTermMemory(BaseMemory):
    def __init__(self, max_size: int = 10):
        self.memory = []
        self.max_size = max_size

    def add(self, text: str):
        self.memory.append(text)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_relevant(self, query: str, k: int = 5) -> List[str]:
        sorted_memory = sorted(self.memory, key=lambda x: simple_similarity(query, x), reverse=True)
        return sorted_memory[:k]

    def clear(self):
        self.memory.clear()