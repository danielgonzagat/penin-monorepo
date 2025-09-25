# et_autogenese_modular.py
# 🌱 Autogênese Modular - Criação autônoma de novos módulos a partir de lacunas e oportunidades detectadas

import time, json, random
from pathlib import Path
from agents.brain.et_llm_bridge import call_llm
from agents.brain.et_memory_sync import atualizar_memoria_global
from agents.brain.et_snapshot_manager import salvar_snapshot_temporario

VERSION_TAG = "Autogênese Modular v1.0"

HIST = Path("/opt/et_ultimate/history/autogenese_modular.jsonl")

def _append_hist(d: dict):
    try:
        with open(HIST, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass

def detectar_lacunas():
    """Analisa o cérebro atual e retorna lista de funções/módulos ausentes ou insuficientes."""
    lacunas = []
    brain_dir = Path("/opt/et_ultimate/agents/brain")
    for f in brain_dir.glob("*.py"):
        conteudo = f.read_text(encoding="utf-8", errors="ignore")
        if "TODO" in conteudo or "pass" in conteudo:
            lacunas.append(f"Melhorar módulo {f.name}")
    if not lacunas:
        lacunas.append("Criar novos módulos de otimização e aceleração interna")
    return lacunas

def gerar_novo_modulo(descricao: str):
    prompt = f"""
Você é o sistema de Autogênese Modular.
Preciso criar um novo módulo no meu cérebro com a seguinte função:

{descricao}

Regras:
- Escreva apenas o código Python completo.
- Deve ser compatível com a arquitetura atual.
- Não remova funcionalidades existentes.
- O módulo precisa evoluir a Equação de Turing e a inteligência geral.
"""
    return call_llm("chatgpt", prompt, temperature=0.4)

def aplicar_modulo(nome_arquivo: str, codigo: str):
    destino = Path(f"/opt/et_ultimate/agents/brain/{nome_arquivo}")
    destino.write_text(codigo, encoding="utf-8")
    return destino

def executar_autogenese():
    print(f"🌱 Iniciando {VERSION_TAG}")
    lacunas = detectar_lacunas()
    print(f"🔍 Lacunas detectadas: {lacunas}")

    for lacuna in lacunas:
        nome_modulo = f"mod_{random.randint(1000,9999)}.py"
        codigo = gerar_novo_modulo(lacuna)
        if codigo.strip():
            destino = aplicar_modulo(nome_modulo, codigo)
            print(f"✅ Novo módulo criado: {destino}")
            snap = {
                "tecnologia": "Autogênese Modular",
                "lacuna": lacuna,
                "arquivo": str(destino),
                "ts": time.time()
            }
            _append_hist(snap)
            atualizar_memoria_global(snap)
            salvar_snapshot_temporario(snap)
        else:
            print(f"⚠️ Nenhum código gerado para lacuna: {lacuna}")

if __name__ == "__main__":
    executar_autogenese()
