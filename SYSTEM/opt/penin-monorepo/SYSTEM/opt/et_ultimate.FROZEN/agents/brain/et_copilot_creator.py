import json
from pathlib import Path
from agents.brain.et_llm_bridge import consultar_ias_fusao_final

COPILOTS_DIR = Path("/opt/et_ultimate/copilotos")

def criar_copiloto(nome: str, objetivo: str, habilidades: list, contexto: dict) -> dict:
    """
    Cria um copiloto virtual com base no objetivo, habilidades e contexto.
    """
    return {
        "nome": nome,
        "objetivo": objetivo,
        "habilidades": habilidades,
        "contexto": contexto
    }

def salvar_copiloto(copiloto: dict):
    """
    Salva o copiloto criado em arquivo JSON individual.
    """
    COPILOTS_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = COPILOTS_DIR / f"{copiloto['nome'].replace(' ', '_')}.json"
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(copiloto, f, ensure_ascii=False, indent=2)

def gerar_copilotos_especializados():
    """
    Gera múltiplos copilotos especializados consultando múltiplas IAs.
    """
    print("🤖 Gerando copilotos especializados via multi-IA...")
    respostas = consultar_ias_fusao_final(
        objetivo="Criar novos copilotos especializados para executar tarefas derivadas do conhecimento mais recente",
        contexto={"fase": "criação_copilotos"}
    )

    if not respostas or "fusao_final" not in respostas:
        print("⚠️ Nenhuma definição de copilotos recebida.")
        return

    copilotos_data = respostas["fusao_final"]
    if isinstance(copilotos_data, str):
        try:
            copilotos_data = json.loads(copilotos_data)
        except Exception:
            copilotos_data = []

    if not isinstance(copilotos_data, list):
        print("⚠️ Estrutura inválida para copilotos.")
        return

    for copiloto_info in copilotos_data:
        nome = copiloto_info.get("nome", "Copiloto_Desconhecido")
        objetivo = copiloto_info.get("objetivo", "Executar tarefa genérica")
        habilidades = copiloto_info.get("habilidades", [])
        copiloto = criar_copiloto(nome, objetivo, habilidades, copiloto_info)
        salvar_copiloto(copiloto)

    print(f"✅ {len(copilotos_data)} copilotos criados e salvos com sucesso.")
