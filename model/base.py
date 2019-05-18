import numpy as np
from typing import List

class ModelBase():

    def fit(self, docs : List[np.ndarray]) -> None:
        raise NotImplementedError

    def transform(self, word : str) -> float:
        raise NotImplementedError