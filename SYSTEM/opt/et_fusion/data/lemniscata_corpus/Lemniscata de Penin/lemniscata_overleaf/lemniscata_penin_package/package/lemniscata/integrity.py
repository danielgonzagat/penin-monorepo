# -*- coding: utf-8 -*-
from typing import Iterable, List

def clamp01(x: float) -> float:
    if x < 0.0: 
        return 0.0
    if x > 1.0: 
        return 1.0
    return x

def I_from_risk(R_total: float, R_max: float) -> float:
    if R_max <= 0.0:
        return 0.0
    I = 1.0 - (R_total / R_max)
    return clamp01(I)

def I_hard_min(values: Iterable[float]) -> float:
    vals = [clamp01(v) for v in values]
    return min(vals) if vals else 0.0

def I_product(values: Iterable[float]) -> float:
    prod = 1.0
    any_val = False
    for v in values:
        any_val = True
        prod *= clamp01(v)
    return prod if any_val else 0.0

def I_weighted(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    s = sum(weights)
    if s <= 0.0:
        return 0.0
    weights = [w / s for w in weights]
    acc = 0.0
    for v, w in zip(values, weights):
        acc += clamp01(v) * w
    return clamp01(acc)
