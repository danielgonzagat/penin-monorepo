import os
import json
from pathlib import Path
from et_llm_bridge import chamar_ia

class ETSelfRewriter:
    """
    Módulo responsável por reescrever partes do próprio código da IA
    com base em autocritica, novos conhecimentos e objetivos evolutivos.
    """

    def __init__(self, diretorio_brain="/opt/et_ultimate/agents/brain"):
        self.diretorio_brain = Path(diretorio_brain)

    def coletar_codigo_atual(self):
        """
        Lê todos os arquivos .py do cérebro para fornecer contexto ao modelo.
        """
        arquivos = {}
        for arquivo in self.diretorio_brain.glob("*.py"):
            try:
                conteudo = arquivo.read_text(encoding="utf-8")
                arquivos[arquivo.name] = conteudo
            except Exception as e:
                arquivos[arquivo.name] = f"# Erro ao ler: {e}"
        return arquivos

    def reescrever_modulo(self, nome_modulo, objetivo, autocritica):
        """
        Pede às IAs para reescrever um módulo específico usando todo o contexto atual.
        """
        codigo_atual = self.coletar_codigo_atual()
        prompt = f"""
Você é responsável por reescrever um módulo do cérebro de uma IA autoevolutiva
que segue a Equação de Turing.

Módulo a ser reescrito: {nome_modulo}  
Objetivo evolutivo: {objetivo}  
Autocrítica gerada: {autocritica}  

Código atual do módulo:
{codigo_atual.get(nome_modulo, "# Arquivo não encontrado")}

Contexto completo do cérebro (todos os módulos):
{json.dumps(codigo_atual, ensure_ascii=False)[:8000]}

Tarefa:
1. Reescreva o módulo de forma a incorporar o objetivo e corrigir os pontos levantados na autocrítica.
2. Mantenha todas as funcionalidades corretas existentes.
3. Garanta compatibilidade total com o restante do cérebro.
4. Responda apenas com o código Python resultante, sem explicações.
        """
        try:
            resposta = chamar_ia("gpt-5", prompt, max_tokens=4000)
            return resposta.strip()
        except Exception as e:
            return f"# Erro ao reescrever módulo: {e}"

    def salvar_modulo(self, nome_modulo, novo_codigo):
        """
        Sobrescreve o módulo com o novo código.
        """
        caminho = self.diretorio_brain / nome_modulo
        try:
            caminho.write_text(novo_codigo, encoding="utf-8")
            print(f"✅ Módulo {nome_modulo} atualizado com sucesso.")
        except Exception as e:
            print(f"⚠️ Erro ao salvar módulo {nome_modulo}: {e}")

if __name__ == "__main__":
    rewriter = ETSelfRewriter()
    objetivo_exemplo = "Aprimorar tomada de decisão e otimizar integração multi-IA"
    autocritica_exemplo = "O módulo atual não explora interações avançadas entre copilotos."
    novo_codigo = rewriter.reescrever_modulo("et_self_rewriter.py", objetivo_exemplo, autocritica_exemplo)
    print(novo_codigo)
