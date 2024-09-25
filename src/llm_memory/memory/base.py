from abc import ABC, abstractmethod
from typing import List

class BaseMemory(ABC):
    @abstractmethod
    def add(self, text: str):
        pass

    @abstractmethod
    def get_relevant(self, query: str, k: int = 5) -> List[str]:
        pass
