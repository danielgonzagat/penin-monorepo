import json
import time
from pathlib import Path
from agents.brain.et_ensemble_fusion import ensemble_fusion
from agents.brain.et_llm_bridge import requisitar_mutacoes

class SinteseMultiverso:
    """
    Módulo de Síntese Multiverso da ETΩ.
    Coleta múltiplas realidades de possíveis soluções, funde seus melhores elementos
    e produz uma única saída otimizada e evolutiva.
    """

    def __init__(self, arquivo_sintese="/opt/et_ultimate/history/sintese_multiverso.jsonl"):
        self.arquivo_sintese = Path(arquivo_sintese)
        self.arquivo_sintese.parent.mkdir(parents=True, exist_ok=True)

    def gerar_multiverso(self, objetivo, n_variantes=5):
        variantes = []
        for _ in range(n_variantes):
            try:
                mutacoes = requisitar_mutacoes(objetivo)
                variantes.extend(mutacoes)
            except Exception as e:
                variantes.append({"erro": str(e)})
        return variantes

    def sintetizar(self, objetivo, n_variantes=5):
        variantes = self.gerar_multiverso(objetivo, n_variantes=n_variantes)
        fusao_final = ensemble_fusion(variantes)
        registro = {
            "timestamp": time.time(),
            "objetivo": objetivo,
            "variantes": variantes,
            "fusao": fusao_final
        }
        with open(self.arquivo_sintese, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return fusao_final

    def recuperar_historico(self, limite=5):
        try:
            linhas = self.arquivo_sintese.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(l) for l in linhas[-limite:]]
        except FileNotFoundError:
            return []

if __name__ == "__main__":
    sm = SinteseMultiverso()
    resultado = sm.sintetizar("Evoluir a Equação de Turing", n_variantes=3)
    print("Fusão final:", resultado)
    print("Histórico:", sm.recuperar_historico())
