from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseStorage(ABC):
    @abstractmethod
    def store(self, text: str, embedding: np.ndarray):
        pass

    @abstractmethod
    def get_all(self) -> List[Tuple[str, np.ndarray]]:
        pass