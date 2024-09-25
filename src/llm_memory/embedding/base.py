from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass