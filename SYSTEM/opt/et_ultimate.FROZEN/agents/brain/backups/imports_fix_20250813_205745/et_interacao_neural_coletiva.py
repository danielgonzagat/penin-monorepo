import time
import json
from pathlib import Path
from et_llm_bridge import requisitar_mutacoes
from et_ensemble_fusion import ensemble_fusion
from et_watchdog import loop_guard

class InteracaoNeuralColetiva:
    """
    Módulo de Interação Neural Coletiva da ETΩ.
    Conecta múltiplas instâncias e IAs integradas para pensar, discutir e evoluir em conjunto,
    permitindo que o conhecimento circule e seja refinado coletivamente antes da aplicação.
    """

    def __init__(self, arquivo_log="/opt/et_ultimate/history/interacao_neural_coletiva.jsonl"):
        self.arquivo_log = Path(arquivo_log)
        self.arquivo_log.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, dados):
        with open(self.arquivo_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(dados, ensure_ascii=False) + "\n")

    def rodada_coletiva(self, topico, ideias):
        print(f"[INC] Rodada coletiva iniciada para tópico: {topico}")
        print(f"[INC] Ideias iniciais: {len(ideias)}")

        mutacoes = []
        for ideia in ideias:
            novas_mutacoes = requisitar_mutacoes(ideia)
            mutacoes.extend([m for m in novas_mutacoes if not loop_guard(m.get("eq", ""))["stuck"]])

        if not mutacoes:
            print("[INC] Nenhuma mutação viável encontrada.")
            return None

        fusao_final = ensemble_fusion(mutacoes)
        registro = {
            "ts": time.time(),
            "topico": topico,
            "ideias": ideias,
            "mutacoes": mutacoes,
            "fusao_final": fusao_final
        }
        self.registrar(registro)
        print("[INC] Fusão coletiva aplicada com sucesso.")
        return fusao_final

    def historico(self, limite=5):
        try:
            linhas = self.arquivo_log.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(l) for l in linhas[-limite:]]
        except FileNotFoundError:
            return []

if __name__ == "__main__":
    inc = InteracaoNeuralColetiva()
    resultado = inc.rodada_coletiva(
        topico="Evolução da Equação de Turing",
        ideias=[
            "Aprimorar arquitetura cognitiva com auto-ajuste",
            "Adicionar camada de previsão de mutações futuras",
            "Integrar avaliação simbólica e numérica em paralelo"
        ]
    )
    if resultado:
        print("Resultado da evolução coletiva:", resultado)
    print("Últimas rodadas:", inc.historico())
