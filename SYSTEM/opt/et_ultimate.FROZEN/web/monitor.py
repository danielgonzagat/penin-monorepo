from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
import json, os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def show_graph():
    path = "/opt/et_ultimate/history/etomega_scores.jsonl"
    if not os.path.exists(path): return "<h1>Sem dados ainda</h1>"
    with open(path) as f:
        data = [json.loads(l) for l in f.readlines()]
    fig = go.Figure()
    ias = list(set(x['ia'] for x in data))
    for ia in ias:
        y = [x['score'] for x in data if x['ia'] == ia]
        fig.add_trace(go.Scatter(y=y, mode='lines+markers', name=ia))
    return fig.to_html()
