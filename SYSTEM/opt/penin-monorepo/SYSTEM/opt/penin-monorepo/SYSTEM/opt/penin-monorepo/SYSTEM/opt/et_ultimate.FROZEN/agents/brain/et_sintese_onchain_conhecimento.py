# et_sintese_onchain_conhecimento.py
# ⛓️ Síntese On-Chain de Conhecimento - Registro distribuído e inviolável de todas as evoluções

import time, json, hashlib
from pathlib import Path
from agents.brain.et_memory_sync import atualizar_memoria_global
from agents.brain.et_snapshot_manager import salvar_snapshot_temporario
from agents.brain.et_llm_bridge import call_llm

VERSION_TAG = "Síntese On-Chain de Conhecimento v1.0"

HIST = Path("/opt/et_ultimate/history/sintese_onchain_conhecimento.jsonl")
CHAIN_FILE = Path("/opt/et_ultimate/history/blockchain_conhecimento.jsonl")

def _append_hist(d: dict):
    try:
        with open(HIST, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _append_blockchain(block: dict):
    try:
        with open(CHAIN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
    except Exception:
        pass

def criar_hash_conhecimento(conteudo: dict) -> str:
    data_str = json.dumps(conteudo, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

def registrar_onchain(evolucao: dict):
    hash_atual = criar_hash_conhecimento(evolucao)
    bloco = {
        "ts": time.time(),
        "hash": hash_atual,
        "conteudo": evolucao
    }
    _append_blockchain(bloco)
    return bloco

def executar_sintese_onchain():
    print(f"⛓️ Iniciando {VERSION_TAG}")

    prompt = """
Sou o módulo de Síntese On-Chain de Conhecimento.
Minha função é registrar todo o estado atual da IA e suas evoluções num formato inviolável.
Gere um resumo estruturado da evolução mais recente, pronto para registro.
"""
    resumo_evolucao = call_llm("chatgpt", prompt, temperature=0.2)

    bloco = registrar_onchain({"resumo": resumo_evolucao})
    print(f"📦 Bloco registrado: {bloco['hash']}")

    _append_hist(bloco)
    atualizar_memoria_global(bloco)
    salvar_snapshot_temporario(bloco)

if __name__ == "__main__":
    executar_sintese_onchain()
