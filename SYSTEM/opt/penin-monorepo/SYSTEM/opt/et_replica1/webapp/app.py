from flask import Flask, request, jsonify, send_file
import json, os
from transformers import pipeline

app = Flask(__name__)
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

@app.route("/")
def index():
    return """
    <h1>ETΩ Painel Interativo</h1>
    <ul>
        <li><a href="/evolucao">📈 Ver gráfico da equação</a></li>
        <li><a href="/perguntar?q=Como melhorar a equação?">🤖 Perguntar ao copiloto</a></li>
        <li><a href="/ultimos">📄 Ver últimos resultados</a></li>
    </ul>
    """

@app.route("/perguntar")
def perguntar():
    q = request.args.get("q", "Explique a equação de Turing.")
    r = llm(q, max_new_tokens=128)[0]["generated_text"]
    return f"<h2>Pergunta:</h2><pre>{q}</pre><h2>Resposta:</h2><pre>{r}</pre>"

@app.route("/evolucao")
def grafico():
    path = "/opt/et_ultimate/workspace/evolution_chart.png"
    return send_file(path, mimetype="image/png")

@app.route("/ultimos")
def ultimos():
    path = "/opt/et_ultimate/actions/results.jsonl"
    if not os.path.exists(path): return "Sem resultados ainda."
    linhas = list(open(path))[-20:]
    return "<h2>Últimos resultados</h2><pre>" + "".join(linhas) + "</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9898)
