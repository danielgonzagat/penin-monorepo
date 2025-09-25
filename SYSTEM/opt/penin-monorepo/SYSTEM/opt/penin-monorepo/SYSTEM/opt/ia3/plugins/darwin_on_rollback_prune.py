#!/usr/bin/env python3
"""
Hook de rollback Darwin → Poda neural
Remove o último neurônio adicionado se houver regressão
"""
import os, json, torch
from pathlib import Path
from datetime import datetime

IA3_HOME = Path(os.getenv("IA3_HOME", "/opt/ia3"))
BRAIN_STATE = IA3_HOME / "brain" / "brain_state.pt"
WORM_LOG = IA3_HOME / "logs" / "worm.log"

def append_worm(event):
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(WORM_LOG, "a") as f:
        f.write("EVENT:" + json.dumps(event, ensure_ascii=False) + "\n")

def main():
    if not BRAIN_STATE.exists():
        append_worm({"event":"rollback_prune_skip","reason":"no_brain_state"})
        print("⚠️ Sem estado do cérebro para rollback")
        return

    data = torch.load(BRAIN_STATE, map_location="cpu")
    H = int(data["hidden_dim"])
    generation = data.get("generation", 0)
    
    if H <= 1:
        append_worm({"event":"rollback_prune_skip","reason":"minimum_neurons"})
        print("⚠️ Cérebro no tamanho mínimo (1 neurônio)")
        return

    print(f"🔄 Rollback: removendo neurônio #{H}")
    
    # "Remove" o último neurônio: recorta matrizes
    newH = H - 1
    data["W_in"] = data["W_in"][:newH, :]
    data["b_h"] = data["b_h"][:newH]
    data["W_hh"] = data["W_hh"][:newH, :newH]
    data["W_out"] = data["W_out"][:, :newH]
    data["hidden_dim"] = newH
    data["generation"] = generation - 1
    
    # Recalcular consciência
    recursion = torch.abs(data["W_hh"]).mean().item()
    size = (newH / 100.0)
    data["consciousness_level"] = min(1.0, recursion * 0.5 + size * 0.5)
    
    torch.save(data, BRAIN_STATE)

    append_worm({
        "event": "rollback_prune_done",
        "hidden_before": H,
        "hidden_after": newH,
        "generation": data["generation"],
        "consciousness": data["consciousness_level"]
    })
    
    print(f"✅ Neurônio removido. Cérebro: {newH} neurônios")

if __name__ == "__main__":
    main()