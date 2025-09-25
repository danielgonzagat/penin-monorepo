#!/usr/bin/env python3
"""
Ponte entre Darwin existente e o novo sistema IA³
Conecta eventos Darwin → Neurogênese
"""
import os, sys, json, subprocess
from pathlib import Path

def integrate_with_darwin():
    """Configura Darwin para usar hooks IA³"""
    
    # Atualizar política Darwin
    darwin_config = {
        "spawn_hook": "/opt/ia3/plugins/darwin_on_spawn_neurogenesis.py",
        "rollback_hook": "/opt/ia3/plugins/darwin_on_rollback_prune.py",
        "deaths_per_birth": 10  # A cada 10 mortes, nasce 1 neurônio
    }
    
    config_path = Path("/root/darwin/darwin_policy.json")
    if config_path.exists():
        with open(config_path) as f:
            policy = json.load(f)
        policy["hooks"] = darwin_config
        with open(config_path, "w") as f:
            json.dump(policy, f, indent=2)
        print("✅ Darwin configurado para IA³")
    else:
        print("⚠️ Config Darwin não encontrada em /root/darwin/darwin_policy.json")
        print("   Usando darwinctl diretamente...")
    
    # Configurar darwinctl
    print("\n=== COMANDOS DARWIN PARA IA³ ===")
    print("# Spawn manual (cria 1 neurônio):")
    print("darwinctl spawn --plugin /opt/ia3/plugins/darwin_on_spawn_neurogenesis.py")
    print("\n# Birth window automático:")
    print("darwinctl birth-window --plugin /opt/ia3/plugins/darwin_on_spawn_neurogenesis.py")
    print("\n# Rollback (remove último neurônio):")
    print("darwinctl rollback --worm /opt/ia3/logs/worm.log")
    
if __name__ == "__main__":
    integrate_with_darwin()