import os, torch
from dataclasses import dataclass

@dataclass
class EngineState:
    cycle: int = 0
    deaths: int = 0
    births: int = 0
    ia3_pass_rate: float = 0.0
    last_loss: float = float("inf")

def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs("/var/log", exist_ok=True)

def save_state(path: str, model, state: EngineState):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "engine": state.__dict__
    }, path)

def load_state(path: str):
    if not os.path.exists(path): return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict): return None
        return blob
    except Exception:
        return None