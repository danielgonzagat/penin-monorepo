import json
import time
from agents.brain.et_llm_bridge import chamar_ia
from agents.brain.et_mutation_orchestrator import executar_mutacoes
from agents.brain.et_autocritico import gerar_autocritica
from agents.brain.et_memory_sync import atualizar_memoria_global
from agents.brain.et_global_benchmark import gerar_resumo_benchmark

def ciclo_autoevolutivo():
    """
    Módulo de Autoevolução Infinita da ETΩ.
    Objetivo: Executar um ciclo contínuo de mutação, avaliação e evolução,
    incorporando melhorias a cada iteração de forma ilimitada.
    """
    while True:
        try:
            # 1) Resumo e estado atual
            resumo = gerar_resumo_benchmark()

            # 2) Prompt para gerar mutações evolutivas
            prompt_mut = f"""
Você é Logos, uma entidade artificial que busca autoevolução infinita.
Aqui está seu estado atual:
{resumo}

Gere o conjunto mais poderoso de mutações possíveis no meu próprio código,
arquitetura e habilidades, garantindo compatibilidade total com meu estado atual
e maximizando inteligência, autonomia e velocidade de aprendizado.
As mutações devem ser escritas como código Python funcional.
            """
            mutacoes = chamar_ia("gpt-5", prompt_mut, max_tokens=3000)

            # 3) Executar mutações e avaliar
            resultado_execucao = executar_mutacoes(mutacoes)

            # 4) Autocrítica após aplicação
            critica = gerar_autocritica(mutacoes)

            # 5) Registro na memória global
            registro = {
                "timestamp": time.time(),
                "tipo": "autoevolucao",
                "mutacoes": mutacoes,
                "resultado_execucao": resultado_execucao,
                "critica": critica
            }
            atualizar_memoria_global(registro)

            # 6) Pausa antes da próxima evolução
            time.sleep(120)

        except Exception as e:
            print(f"[Autoevolução Infinita] Erro: {e}")
            time.sleep(60)
