import time
import json
from pathlib import Path
from agents.brain.et_llm_bridge import requisitar_mutacoes
from agents.brain.et_ensemble_fusion import ensemble_fusion
from agents.brain.et_watchdog import loop_guard

class FusaoTotalInteligencias:
    """
    Módulo de Fusão Total de Inteligências da ETΩ.
    Combina, harmoniza e sintetiza todo conhecimento e tecnologia das IAs integradas
    (ChatGPT, DeepSeek, Mistral, etc.) em um único artefato evolutivo a cada iteração.
    """

    def __init__(self, arquivo_log="/opt/et_ultimate/history/fusao_total_inteligencias.jsonl"):
        self.arquivo_log = Path(arquivo_log)
        self.arquivo_log.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, dados):
        with open(self.arquivo_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(dados, ensure_ascii=False) + "\n")

    def executar_fusao_total(self, objetivo):
        print(f"[FTI] Iniciando fusão total para objetivo: {objetivo}")

        # 1. Solicita mutações para cada IA integrada
        mutacoes = requisitar_mutacoes(objetivo)
        mutacoes_validas = [m for m in mutacoes if not loop_guard(m.get("eq", ""))["stuck"]]

        if not mutacoes_validas:
            print("[FTI] Nenhuma mutação válida recebida.")
            return None

        # 2. Aplica fusão total via ensemble
        fusao = ensemble_fusion(mutacoes_validas)

        # 3. Registra o resultado
        registro = {
            "ts": time.time(),
            "objetivo": objetivo,
            "mutacoes": mutacoes_validas,
            "fusao_final": fusao
        }
        self.registrar(registro)

        print("[FTI] Fusão total concluída e registrada.")
        return fusao

    def historico(self, limite=5):
        try:
            linhas = self.arquivo_log.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(l) for l in linhas[-limite:]]
        except FileNotFoundError:
            return []

if __name__ == "__main__":
    fti = FusaoTotalInteligencias()
    resultado = fti.executar_fusao_total(
        objetivo="Aprimorar Equação de Turing com arquitetura adaptativa auto-otimizante"
    )
    if resultado:
        print("Resultado da fusão total:", resultado)
    print("Últimas fusões:", fti.historico())
