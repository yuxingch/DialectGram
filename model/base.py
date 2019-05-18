import numpy as np
from typing import Iterable, Any

class ModelBase():

    def fit(self, docs : Iterable[Any]) -> None:
        raise NotImplementedError

    def transform(self, word : str) -> float:
        raise NotImplementedError