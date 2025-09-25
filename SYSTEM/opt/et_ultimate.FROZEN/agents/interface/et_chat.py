# et_chat.py
# Interface simples de chat com a ETΩ via terminal (modo REPL)
import readline
import sys
sys.path.append("/opt/et_ultimate/agents/brain")
from et_llm_bridge import requisitar_mutacoes

print("🧠 Bem-vindo ao Chat da ETΩ (pressione Ctrl+C para sair)\n")

while True:
    try:
        entrada = input("💬 Você: ").strip()
        if not entrada:
            continue
        respostas = requisitar_mutacoes(entrada)
        if not respostas:
            print("⚠️ Nenhuma IA respondeu.")
        else:
            for r in respostas:
                print(f"🤖 {r['ia']}: {r['eq']}\n")
    except KeyboardInterrupt:
        print("\n🚪 Encerrando sessão do Chat da ETΩ...")
        break
    except Exception as e:
        print(f"❌ Erro: {e}")
