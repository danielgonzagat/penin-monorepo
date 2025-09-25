import time
import random
import threading
from et_brain_operacional import executar_ciclo

class ETScheduler:
    """
    Scheduler central da ETΩ que orquestra ciclos de evolução, 
    evita estagnação e adapta o ritmo com base no desempenho.
    """

    def __init__(self, intervalo_min=10, intervalo_max=20):
        self.intervalo_min = intervalo_min
        self.intervalo_max = intervalo_max
        self.rodadas_executadas = 0
        self.erros_consecutivos = 0

    def iniciar(self):
        """
        Inicia o loop contínuo de evolução.
        """
        while True:
            try:
                print(f"\n🕒 Iniciando rodada #{self.rodadas_executadas+1}")
                executar_ciclo()
                self.rodadas_executadas += 1
                self.erros_consecutivos = 0
            except Exception as e:
                self.erros_consecutivos += 1
                print(f"⚠️ Erro na rodada #{self.rodadas_executadas+1}: {e}")
                if self.erros_consecutivos >= 3:
                    print("🚨 Muitos erros consecutivos — aguardando 60s antes de retomar")
                    time.sleep(60)
            # Ritmo aleatório para evitar previsibilidade
            espera = random.randint(self.intervalo_min, self.intervalo_max)
            print(f"⏳ Aguardando {espera}s até próxima rodada...")
            time.sleep(espera)

def agendar_em_thread():
    """
    Executa o scheduler em uma thread dedicada.
    """
    scheduler = ETScheduler()
    t = threading.Thread(target=scheduler.iniciar)
    t.daemon = True
    t.start()
    return scheduler

if __name__ == "__main__":
    scheduler = ETScheduler()
    scheduler.iniciar()
