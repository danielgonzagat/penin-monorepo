import json
from pathlib import Path
from datetime import datetime

HIST_DIR = Path("/opt/et_ultimate/history")
HIST_DIR.mkdir(parents=True, exist_ok=True)

MUTATION_LOG = HIST_DIR / "mutations.jsonl"
MUTATION_TXT = HIST_DIR / "mutations_readable.txt"

def _log(msg):
    ts = datetime.utcnow().isoformat()
    print(f"[MUTACAO_WRITER] {msg}")
    with open(HIST_DIR / "mutacao_writer.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def registrar_mutacao(mutation):
    """
    Registra uma mutação completa no formato JSONL e em formato legível.
    """
    try:
        with open(MUTATION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(mutation, ensure_ascii=False) + "\n")
        with open(MUTATION_TXT, "a", encoding="utf-8") as f:
            f.write(f"[{mutation.get('ts', 'sem_timestamp')}] IA={mutation.get('ia')} | Score={mutation.get('score')} | Novelty={mutation.get('novelty')}\n")
            f.write(f"Objetivo: {mutation.get('objetivo')}\n")
            f.write(f"Equação: {mutation.get('eq')}\n")
            f.write(f"Estratégia: {mutation.get('estrategia')}\n")
            f.write(f"Autocrítica: {mutation.get('autocritica')}\n")
            f.write("-" * 80 + "\n")
        _log(f"✅ Mutação registrada: IA={mutation.get('ia')} Score={mutation.get('score')}")
    except Exception as e:
        _log(f"⚠️ Falha ao registrar mutação: {e}")

def carregar_mutacoes():
    """
    Carrega todas as mutações registradas em formato de lista.
    """
    try:
        if not MUTATION_LOG.exists():
            return []
        with open(MUTATION_LOG, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        _log(f"⚠️ Erro ao carregar mutações: {e}")
    return []

def buscar_mutacoes_por_objetivo(objetivo):
    """
    Retorna mutações filtradas por objetivo.
    """
    try:
        mutacoes = carregar_mutacoes()
        return [m for m in mutacoes if m.get("objetivo") == objetivo]
    except Exception as e:
        _log(f"⚠️ Erro ao buscar mutações por objetivo: {e}")
        return []

def limpar_mutacoes_antigas(limite=500):
    """
    Mantém apenas as últimas 'limite' mutações registradas.
    """
    try:
        mutacoes = carregar_mutacoes()
        if len(mutacoes) > limite:
            mutacoes = mutacoes[-limite:]
            with open(MUTATION_LOG, "w", encoding="utf-8") as f:
                for m in mutacoes:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            _log(f"🧹 Mutações antigas limpas, mantendo {limite} registros.")
    except Exception as e:
        _log(f"⚠️ Erro ao limpar mutações antigas: {e}")
