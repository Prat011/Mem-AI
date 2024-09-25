from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..storage.base import BaseStorage

class BaseRetrieval(ABC):
    @abstractmethod
    def retrieve(self, query_embedding: np.ndarray, storage: BaseStorage, k: int) -> List[str]:
        pass
