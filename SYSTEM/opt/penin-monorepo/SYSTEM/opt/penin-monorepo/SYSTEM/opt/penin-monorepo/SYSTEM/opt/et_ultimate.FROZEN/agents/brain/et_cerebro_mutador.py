import json
from pathlib import Path
from agents.brain.et_llm_bridge import consultar_ias_fusao_final

HIST_DIR = Path("/opt/et_ultimate/history/mutacoes_aplicadas.jsonl")

def aplicar_mutacao(codigo_atual: str, mutacao: str) -> str:
    """
    Aplica a mutação ao código atual, retornando o novo código.
    """
    return mutacao if mutacao.strip() else codigo_atual

def salvar_mutacao(info: dict):
    """
    Salva informações da mutação aplicada para histórico.
    """
    with open(HIST_DIR, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")

def cerebro_mutador(codigo_atual: str, contexto: dict) -> str:
    """
    Orquestra a aplicação de mutações no cérebro:
    - Consulta múltiplas IAs
    - Funde as respostas
    - Aplica a mutação resultante
    - Salva histórico
    """
    print("🧬 Solicitando mutações multi-IA para evolução interna...")
    respostas = consultar_ias_fusao_final(
        objetivo="Gerar mutações internas que melhorem desempenho, autonomia e inteligência",
        contexto=contexto,
        codigo_atual=codigo_atual
    )

    if not respostas or not isinstance(respostas, dict) or "fusao_final" not in respostas:
        print("⚠️ Nenhuma mutação viável recebida.")
        return codigo_atual

    mutacao_final = respostas["fusao_final"]

    codigo_novo = aplicar_mutacao(codigo_atual, mutacao_final)

    salvar_mutacao({
        "contexto": contexto,
        "codigo_antigo": codigo_atual,
        "mutacao": mutacao_final,
        "codigo_novo": codigo_novo
    })

    print("✅ Mutação aplicada e salva no histórico.")
    return codigo_novo
