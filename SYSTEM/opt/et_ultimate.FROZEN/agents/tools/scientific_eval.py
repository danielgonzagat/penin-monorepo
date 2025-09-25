from sympy import simplify

def scientific_score(eq_str):
    try:
        simplified = simplify(eq_str)
        return 100 - len(str(simplified))
    except Exception:
        return 0
