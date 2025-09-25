# -*- coding: utf-8 -*-
from typing import List

class MinMax:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.vmin = None
        self.vmax = None

    def fit(self, arr: List[float]):
        self.vmin = float(min(arr))
        self.vmax = float(max(arr))

    def transform(self, arr: List[float]) -> List[float]:
        if self.vmin is None or self.vmax is None:
            raise RuntimeError("MinMax not fitted.")
        denom = max(self.eps, self.vmax - self.vmin)
        return [(x - self.vmin) / denom for x in arr]

    def fit_transform(self, arr: List[float]) -> List[float]:
        self.fit(arr)
        return self.transform(arr)
