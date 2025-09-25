import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SNAPSHOT_DIR = Path("/opt/et_ultimate/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

def criar_snapshot(tag: Optional[str] = None) -> Path:
    """
    Cria um snapshot completo do estado atual do cérebro e salva em disco.
    Pode receber um 'tag' opcional para identificar o snapshot.
    Retorna o caminho do snapshot criado.
    """
    estado: Dict[str, Any] = {}
    try:
        for nome, valor in globals().items():
            try:
                estado[nome] = str(valor)
            except Exception:
                estado[nome] = "<não serializável>"

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"snapshot_{tag+'_' if tag else ''}{timestamp}.json"
        caminho = SNAPSHOT_DIR / filename

        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)

        return caminho
    except Exception as e:
        raise RuntimeError(f"Falha ao criar snapshot: {e}")

def restaurar_ultimo_snapshot() -> Optional[Dict[str, Any]]:
    """
    Restaura o último snapshot salvo e retorna seu conteúdo como dicionário.
    Caso não exista snapshot, retorna None.
    """
    try:
        snapshots = sorted(SNAPSHOT_DIR.glob("snapshot_*.json"), key=os.path.getmtime, reverse=True)
        if not snapshots:
            return None

        ultimo = snapshots[0]
        with open(ultimo, "r", encoding="utf-8") as f:
            dados = json.load(f)
        return dados
    except Exception as e:
        raise RuntimeError(f"Falha ao restaurar snapshot: {e}")
