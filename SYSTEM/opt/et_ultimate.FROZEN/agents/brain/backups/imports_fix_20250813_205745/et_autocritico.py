import json
import time
from pathlib import Path
from et_llm_bridge import consultar_ias_fusao_final
from et_memory_sync import atualizar_memoria_global

HISTORICO_AUTOCRITICA = Path("/opt/et_ultimate/history/autocriticas.jsonl")

PERGUNTAS_BASE = [
    "Como isso melhora minha estrutura?",
    "Como isso pode me deixar mais rápido?",
    "Como isso pode me deixar mais inteligente?",
    "Como isso pode me deixar mais consciente?",
    "Como isso pode melhorar minha autocrítica?",
    "Como isso pode me levar a criticar coisas que não estou criticando?",
    "Como eu posso superar minhas próprias limitações com essa novidade?",
    "Como eu posso ganhar mais habilidade com isso?",
    "Como eu posso tornar meu aprendizado mais rápido, eficaz e refinado?",
    "Que novas funções .py posso adicionar no meu cérebro graças a isso?",
    "Que novas tecnologias posso incorporar graças a isso?",
    "Que novas habilidades, tecnologias, módulos e melhorias posso traduzir para .py e integrar no meu cérebro?"
]

PERGUNTA_EXTRA = (
    "Que outras perguntas eu poderia ter feito que resultariam em evoluções em .py "
    "para adicionar ou criar novos arquivos no meu cérebro, me levando a me tornar a IA mais inteligente da história?"
)

def gerar_autocritica(conhecimento_adquirido: str, estado_cerebro: dict) -> dict:
    """
    Executa o processo completo de autocrítica evolutiva, conversando com todas as IAs,
    fundindo as respostas e produzindo um plano de evolução concreto.
    """
    print("🧠 [Autocrítico] Iniciando análise de evolução...")
    perguntas = [f"{p} Conhecimento adquirido: {conhecimento_adquirido}" for p in PERGUNTAS_BASE]

    # 1) Consultar todas as IAs para cada pergunta
    respostas = {}
    for pergunta in perguntas:
        respostas[pergunta] = consultar_ias_fusao_final(pergunta, timeout_seg=600, tentativas=3)

    # 2) Pergunta extra para gerar novas perguntas potenciais
    novas_perguntas = consultar_ias_fusao_final(PERGUNTA_EXTRA, timeout_seg=600, tentativas=3)
    respostas["Novas perguntas sugeridas"] = novas_perguntas

    # 3) Arbitrar e gerar um único plano final com ChatGPT
    prompt_arbitro = {
        "objetivo": "Criar a evolução mais impactante possível respeitando a Equação de Turing",
        "estado_cerebro": estado_cerebro,
        "conhecimento": conhecimento_adquirido,
        "perguntas": PERGUNTAS_BASE,
        "respostas": respostas,
        "novas_perguntas": novas_perguntas,
        "restricoes": [
            "Nada será perdido",
            "Nenhum código será removido",
            "Apenas criar novas funções ou evoluir as já existentes",
            "A Equação de Turing será integralmente respeitada e evoluída"
        ]
    }

    codigo_final = consultar_ias_fusao_final(
        f"Essas são todas as informações para criar minha evolução final: {json.dumps(prompt_arbitro, ensure_ascii=False)}. "
        f"Escreva apenas o código Python final da evolução e todos os comandos de teste de sintaxe.",
        timeout_seg=None,  # sem limite para a fusão final
        tentativas=1
    )

    # 4) Registrar histórico
    registro = {
        "ts": time.time(),
        "conhecimento": conhecimento_adquirido,
        "plano_evolucao": codigo_final
    }
    HISTORICO_AUTOCRITICA.write_text(
        (HISTORICO_AUTOCRITICA.read_text(encoding="utf-8") if HISTORICO_AUTOCRITICA.exists() else "") +
        json.dumps(registro, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # 5) Atualizar memória global
    atualizar_memoria_global({"autocritica": registro})

    print("✅ [Autocrítico] Evolução gerada e registrada.")
    return registro
