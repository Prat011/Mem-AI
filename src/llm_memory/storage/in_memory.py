from .base import BaseStorage
from typing import List, Tuple
import numpy as np

class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.data = []

    def store(self, text: str, embedding: np.ndarray):
        self.data.append((text, embedding))

    def get_all(self) -> List[Tuple[str, np.ndarray]]:
        return self.data