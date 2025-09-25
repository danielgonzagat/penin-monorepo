import json
import time
from pathlib import Path
from et_memory_sync import atualizar_memoria_global, buscar_similares

class MemoriaHipercontextual:
    """
    Módulo de Memória Hipercontextual da ETΩ.
    Permite registrar, buscar e recuperar conhecimento com máxima profundidade semântica,
    preservando conexões entre contexto passado, presente e projeções futuras.
    """

    def __init__(self, arquivo_memoria="/opt/et_ultimate/history/memoria_hipercontextual.jsonl"):
        self.arquivo_memoria = Path(arquivo_memoria)
        self.arquivo_memoria.parent.mkdir(parents=True, exist_ok=True)

    def registrar(self, conteudo, tags=None, origem=None):
        registro = {
            "timestamp": time.time(),
            "conteudo": conteudo,
            "tags": tags or [],
            "origem": origem
        }
        with open(self.arquivo_memoria, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        atualizar_memoria_global(registro)

    def buscar(self, consulta, limite=5):
        try:
            similares = buscar_similares(consulta, k=limite)
            return similares
        except Exception as e:
            return [{"erro": str(e)}]

    def recuperar_contexto(self, topicos):
        contexto_completo = []
        for topico in topicos:
            resultados = self.buscar(topico, limite=10)
            contexto_completo.extend(resultados)
        return contexto_completo

    def consolidar_para_prompt(self, topicos):
        contexto = self.recuperar_contexto(topicos)
        texto = "\n".join([json.dumps(c, ensure_ascii=False) for c in contexto])
        return f"=== CONTEXTO HIPERCONTEXTUAL ===\n{texto}\n=== FIM CONTEXTO ==="


if __name__ == "__main__":
    mhc = MemoriaHipercontextual()
    mhc.registrar("Primeiro registro de teste", tags=["teste", "inicio"], origem="manual")
    print(mhc.buscar("teste"))
    print(mhc.consolidar_para_prompt(["teste"]))
