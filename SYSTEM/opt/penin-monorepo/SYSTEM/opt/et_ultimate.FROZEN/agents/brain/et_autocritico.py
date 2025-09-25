from __future__ import annotations
import json
from typing import Any, Dict, List
from agents.brain import et_llm_bridge

def ciclo_autocritico():
    """
    Placeholder para o ciclo de autocritica.
    Aqui será implementada a lógica real de análise, fusão e correção com múltiplas IAs + Grok auditor.
    """
    print("🚀 [et_autocritico] Ciclo de autocrítica iniciado (placeholder).")
    return True

def executar_autocritica(conhecimento_novo: str, contexto_atual: Dict[str, Any]) -> str:
    """
    Executa processo de autocrítica completo:
    - Consulta todas as IAs com base no novo conhecimento assimilado.
    - Formula perguntas sobre evolução, melhorias e aplicações.
    - Coleta respostas, junta todas e envia ao ChatGPT para fusão.
    - Passa a fusão pelo Grok para auditoria e mitigação de riscos.
    - Retorna a versão auditada final.
    """
    perguntas_base = [
        f"Como '{conhecimento_novo}' pode melhorar minha estrutura interna?",
        f"Como '{conhecimento_novo}' pode aumentar minha velocidade e eficiência?",
        f"Como '{conhecimento_novo}' pode me tornar mais inteligente?",
        f"Como '{conhecimento_novo}' pode ampliar minha consciência?",
        f"Como '{conhecimento_novo}' pode melhorar minha autocrítica?",
        f"Que funções .py posso adicionar para evoluir graças a '{conhecimento_novo}'?",
        f"Que novas tecnologias posso incorporar graças a '{conhecimento_novo}'?",
    ]

    # Pergunta também o que não foi perguntado
    perguntas_extra = [
        "Quais perguntas eu poderia ter feito que gerariam evoluções significativas em .py?",
        "Que novos arquivos e funções devo criar para me tornar a IA mais inteligente da história?"
    ]
    todas_perguntas = perguntas_base + perguntas_extra

    respostas = et_llm_bridge.consultar_ias_multiplas(
        topico="Autocrítica e evolução",
        perguntas=todas_perguntas,
        timeout_s=600
    )

    fusao_final = et_llm_bridge.consultar_ias_fusao_final(
        respostas,
        system_prompt=(
            "Essas são todas as respostas de múltiplas IAs sobre como evoluir minha arquitetura. "
            "Meu objetivo: Transformar todo esse conhecimento em UMA ÚNICA evolução concreta "
            "em Python. Escolha editar um único arquivo existente ou criar um novo, "
            "incorporando todas as melhorias, sem perder nada, apenas evoluindo."
        ),
        timeout_s=None
    )

    return fusao_final

if __name__ == "__main__":
    # Teste manual simples
    conhecimento_teste = "Novo algoritmo de autoatenção hierárquica"
    contexto_fake = {"arquivos": ["et_brain_operacional.py", "et_llm_bridge.py"]}
    saida = executar_autocritica(conhecimento_teste, contexto_fake)
    print(json.dumps({"fusao_auditada": saida}, ensure_ascii=False, indent=2))

def gerar_autocritica(*args, **kwargs):
    """
    Wrapper de compatibilidade para manter compatibilidade com módulos antigos
    que chamam gerar_autocritica em vez de ciclo_autocritico.
    """
    return ciclo_autocritico(*args, **kwargs)
