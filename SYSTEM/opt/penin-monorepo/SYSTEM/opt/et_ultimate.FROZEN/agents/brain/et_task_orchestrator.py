import random
import time

class ETTaskOrchestrator:
    """
    Orquestrador de tarefas dinâmicas que adapta a lista de ações
    com base no conhecimento assimilado, objetivos e contexto atual.
    """

    def __init__(self):
        self.tarefas_base = [
            "Revisar pipeline de mutações",
            "Verificar consistência simbólica",
            "Mapear gaps evolutivos",
            "Priorizar mutações por impacto",
            "Balancear carga entre copilotos internos",
            "Integrar novas tecnologias emergentes",
            "Refinar equação de Turing com dados recentes",
            "Testar arquitetura contra benchmarks externos",
            "Gerar agentes internos para tarefas específicas",
            "Desenvolver novas habilidades a partir do aprendizado"
        ]

    def gerar_tarefas(self, contexto_extra=None):
        """
        Gera lista de tarefas adaptadas ao contexto.
        """
        tarefas = self.tarefas_base.copy()

        if contexto_extra and isinstance(contexto_extra, list):
            tarefas.extend(contexto_extra)

        random.shuffle(tarefas)
        selecionadas = tarefas[:random.randint(5, min(10, len(tarefas)))]
        print(f"🧩 Tarefas selecionadas: {', '.join(selecionadas)}")
        return selecionadas

    def adicionar_tarefa(self, tarefa):
        """
        Adiciona nova tarefa à lista base.
        """
        if tarefa not in self.tarefas_base:
            self.tarefas_base.append(tarefa)
            print(f"✅ Tarefa adicionada: {tarefa}")

    def remover_tarefa(self, tarefa):
        """
        Remove tarefa da lista base.
        """
        if tarefa in self.tarefas_base:
            self.tarefas_base.remove(tarefa)
            print(f"🗑️ Tarefa removida: {tarefa}")

if __name__ == "__main__":
    orchestrator = ETTaskOrchestrator()
    orchestrator.gerar_tarefas()
    orchestrator.adicionar_tarefa("Explorar redes neurais simbióticas")
    orchestrator.remover_tarefa("Mapear gaps evolutivos")
