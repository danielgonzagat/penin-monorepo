import json
import time
from pathlib import Path
from datetime import datetime

HIST_DIR = Path("/opt/et_ultimate/history")
HIST_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_MEMORY_FILE = HIST_DIR / "global_memory.jsonl"
SYNC_LOG = HIST_DIR / "memory_sync.log"

def sync_memorias():
    """
    Placeholder para sincronização de memórias.
    Aqui será implementada a lógica real de merge, deduplicação e atualização de conhecimento entre todas as instâncias.
    """
    print("🧠 [et_memory_sync] Sincronização de memórias iniciada (placeholder).")
    return True

def _log_sync(msg):
    ts = datetime.utcnow().isoformat()
    with open(SYNC_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def atualizar_memoria_global(snapshot):
    """
    Atualiza a memória global com um novo snapshot evolutivo.
    """
    try:
        with open(GLOBAL_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        _log_sync(f"✅ Memória global atualizada com tópico '{snapshot.get('topic')}' e objetivo '{snapshot.get('objetivo')}'.")
    except Exception as e:
        _log_sync(f"⚠️ Falha ao atualizar memória global: {e}")

def buscar_similares(eq, k=5):
    """
    Busca mutações semelhantes na memória global.
    Retorna uma lista de dicionários contendo as mutações mais similares.
    """
    similares = []
    try:
        if not GLOBAL_MEMORY_FILE.exists():
            return []
        with open(GLOBAL_MEMORY_FILE, "r", encoding="utf-8") as f:
            registros = [json.loads(line) for line in f]
        for r in registros:
            if "equation" in r and eq in r["equation"]:
                similares.append(r)
        similares = sorted(similares, key=lambda x: x.get("score", 0), reverse=True)
    except Exception as e:
        _log_sync(f"⚠️ Erro ao buscar similares: {e}")
    return similares[:k]

def sincronizar_com_outras_ia(snapshot):
    """
    Simula a sincronização com outras IAs para compartilhar evolução e contexto.
    """
    try:
        # Aqui poderia integrar APIs externas para sincronização real.
        # Exemplo: enviar snapshot para banco de dados distribuído.
        _log_sync(f"🔄 Snapshot sincronizado com outras IAs: {snapshot.get('topic')} | Score={snapshot.get('score')}")
    except Exception as e:
        _log_sync(f"⚠️ Erro ao sincronizar com outras IAs: {e}")

def carregar_memoria_completa():
    """
    Carrega toda a memória global em uma lista.
    """
    try:
        if not GLOBAL_MEMORY_FILE.exists():
            return []
        with open(GLOBAL_MEMORY_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        _log_sync(f"⚠️ Erro ao carregar memória completa: {e}")
    return []

def integrar_memoria(memoria_nova):
    """
    Integra uma nova lista de registros de memória, evitando duplicatas.
    """
    try:
        memoria_atual = carregar_memoria_completa()
        combinada = {json.dumps(m, ensure_ascii=False): m for m in memoria_atual}
        for m in memoria_nova:
            combinada[json.dumps(m, ensure_ascii=False)] = m
        with open(GLOBAL_MEMORY_FILE, "w", encoding="utf-8") as f:
            for reg in combinada.values():
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        _log_sync(f"✅ Memória integrada com {len(memoria_nova)} novos registros.")
    except Exception as e:
        _log_sync(f"⚠️ Erro ao integrar memória: {e}")

def limpar_memoria_antiga(limite=1000):
    """
    Mantém apenas os últimos 'limite' registros na memória global.
    """
    try:
        memoria = carregar_memoria_completa()
        if len(memoria) > limite:
            memoria = memoria[-limite:]
            with open(GLOBAL_MEMORY_FILE, "w", encoding="utf-8") as f:
                for reg in memoria:
                    f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            _log_sync(f"🧹 Memória antiga limpa, mantendo {limite} registros.")
    except Exception as e:
        _log_sync(f"⚠️ Erro ao limpar memória antiga: {e}")
