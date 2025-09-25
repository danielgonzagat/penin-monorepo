import time
import json
from pathlib import Path
from et_llm_bridge import requisitar_mutacoes
from et_fusionator import fundir_mutacoes
from et_watchdog import loop_guard

class AdaptacaoMetaorganica:
    """
    Módulo de Adaptação Metaorgânica da ETΩ.
    Inspira-se em sistemas biológicos para adaptar dinamicamente a arquitetura interna da IA,
    reorganizando funções, fluxos e interações para maximizar evolução e inteligência.
    """

    def __init__(self, arquivo_estado="/opt/et_ultimate/history/adaptacao_metaorganica.json"):
        self.arquivo_estado = Path(arquivo_estado)
        self.arquivo_estado.parent.mkdir(parents=True, exist_ok=True)

    def salvar_estado(self, estado):
        with open(self.arquivo_estado, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)

    def carregar_estado(self):
        if self.arquivo_estado.exists():
            return json.loads(self.arquivo_estado.read_text(encoding="utf-8"))
        return {"historico": []}

    def executar_adaptacao(self, objetivo):
        print(f"[AMO] Iniciando adaptação metaorgânica para objetivo: {objetivo}")

        # 1. Solicita mutações que simulam reorganização estrutural
        mutacoes = requisitar_mutacoes(f"Reorganizar arquitetura para: {objetivo}")
        mutacoes_validas = [m for m in mutacoes if not loop_guard(m.get("eq", ""))["stuck"]]

        if not mutacoes_validas:
            print("[AMO] Nenhuma mutação válida recebida.")
            return None

        # 2. Funde mutações em um plano adaptativo
        plano_adaptativo = fundir_mutacoes(mutacoes_validas)

        # 3. Atualiza estado
        estado = self.carregar_estado()
        registro = {
            "ts": time.time(),
            "objetivo": objetivo,
            "plano_adaptativo": plano_adaptativo
        }
        estado["historico"].append(registro)
        self.salvar_estado(estado)

        print("[AMO] Adaptação metaorgânica concluída e registrada.")
        return plano_adaptativo

    def historico(self, limite=5):
        estado = self.carregar_estado()
        return estado["historico"][-limite:]

if __name__ == "__main__":
    amo = AdaptacaoMetaorganica()
    resultado = amo.executar_adaptacao(
        objetivo="Maximizar plasticidade cognitiva da Equação de Turing"
    )
    if resultado:
        print("Plano adaptativo gerado:", resultado)
    print("Últimas adaptações:", amo.historico())
