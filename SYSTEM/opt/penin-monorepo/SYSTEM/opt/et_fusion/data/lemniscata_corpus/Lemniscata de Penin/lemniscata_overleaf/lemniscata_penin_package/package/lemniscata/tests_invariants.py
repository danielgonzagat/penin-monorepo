# -*- coding: utf-8 -*-
import random
from .core import lemniscata, invariants_hold

def run_invariants(n: int = 1000) -> dict:
    ok = 0
    fails = 0
    for _ in range(n):
        E = random.random()
        N = random.random()
        I = random.random()
        mode = random.choice(["partial", "hard"])
        if invariants_hold(E, N, I, mode):
            ok += 1
        else:
            fails += 1
    return {"total": n, "ok": ok, "fails": fails}

if __name__ == "__main__":
    print(run_invariants(2000))
