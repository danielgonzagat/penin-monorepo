#!/usr/bin/env python3
import time, json, os
from pathlib import Path
from prometheus_client import start_http_server, Gauge

IA3_HOME = Path(os.getenv("IA3_HOME", "/opt/ia3"))
LOGS_DIR  = IA3_HOME / "logs"
METRICS_FILE = LOGS_DIR / "metrics.json"

# Métricas Prometheus
g_neurons = Gauge("darwin_brain_neurons_total", "Total de neurônios no cérebro IA³")
g_generation = Gauge("darwin_brain_generation", "Geração atual do cérebro")
g_loss = Gauge("darwin_brain_last_train_loss", "Última loss do treino")
g_avg_loss = Gauge("darwin_brain_avg_train_loss", "Loss média do último treino")
g_consciousness = Gauge("darwin_brain_consciousness", "Nível de consciência estimado (0-1)")

def load_metrics():
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except:
        return {}

if __name__ == "__main__":
    # Porta 9093 para não conflitar com outros exporters
    port = 9093
    start_http_server(port)
    print(f"✅ IA³ Metrics Exporter iniciado na porta {port}")
    
    while True:
        m = load_metrics()
        
        if "darwin_brain_neurons_total" in m:
            g_neurons.set(float(m["darwin_brain_neurons_total"]))
        if "darwin_brain_generation" in m:
            g_generation.set(float(m["darwin_brain_generation"]))
        if "darwin_brain_last_train_loss" in m:
            g_loss.set(float(m["darwin_brain_last_train_loss"]))
        if "darwin_brain_avg_train_loss" in m:
            g_avg_loss.set(float(m["darwin_brain_avg_train_loss"]))
        if "darwin_brain_consciousness" in m:
            g_consciousness.set(float(m["darwin_brain_consciousness"]))
        
        time.sleep(5)