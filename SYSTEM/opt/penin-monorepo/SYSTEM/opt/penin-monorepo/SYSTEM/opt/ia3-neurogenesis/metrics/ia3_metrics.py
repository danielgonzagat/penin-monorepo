#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métricas Prometheus para IA³/DARWIN (CPU) - Patch Avançado v2
- Expõe loss, normas e entropia da ativação
- Usa Histogram (distribuição) + Gauge (último valor) + Summary (observações)
- Histogramas avançados para análise de distribuições
Refs:
- Prometheus client_python: start_http_server e export via HTTP
"""
import os
import time
from typing import Dict, Any

try:
    from prometheus_client import start_http_server, Gauge, Histogram, Summary, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════════

if PROMETHEUS_AVAILABLE:
    # Gauges (valores instantâneos)
    _loss_gauge = Gauge("ia3_loss_last", "Última perda (MSE)")
    _grad_gauge = Gauge("ia3_grad_l2_norm", "Norma L2 do gradiente")
    _weight_gauge = Gauge("ia3_weight_l2_norm", "Norma L2 dos pesos")
    _neurons_gauge = Gauge("ia3_neurons_current", "Número atual de neurônios")
    _consciousness_gauge = Gauge("ia3_consciousness_level", "Nível de consciência")
    
    # Ativação mista
    _act_entropy_gauge = Gauge("ia3_activation_entropy", "Entropia da mistura de ativações")
    _act_choice_gauge = Gauge("ia3_activation_choice_index", "Índice da ativação dominante")
    _act_weight_gauge = Gauge("ia3_activation_weight", "Peso da ativação dominante")
    
    # Counters (acumulativos)
    _births_counter = Counter("ia3_births_total", "Total de neurônios nascidos")
    _deaths_counter = Counter("ia3_deaths_total", "Total de neurônios mortos")
    _cycles_counter = Counter("ia3_cycles_total", "Total de ciclos executados")
    _adaptations_counter = Counter("ia3_adaptations_total", "Total de adaptações bem-sucedidas")
    
    # Histogramas (distribuições)
    _loss_histogram = Histogram(
        "ia3_loss_hist", 
        "Histograma de perdas (MSE)",
        buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    )
    
    _grad_histogram = Histogram(
        "ia3_grad_norm_hist",
        "Histograma de normas de gradiente",
        buckets=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    )
    
    _weight_histogram = Histogram(
        "ia3_weight_norm_hist",
        "Histograma de normas de pesos",
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    )
    
    _consciousness_histogram = Histogram(
        "ia3_consciousness_hist",
        "Histograma de níveis de consciência",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )
    
    # Summary (latência/duração)
    _training_summary = Summary("ia3_training_seconds", "Tempo de treino por ciclo")
    _evaluation_summary = Summary("ia3_evaluation_seconds", "Tempo de avaliação IA³")

else:
    # Dummies se Prometheus não disponível
    class DummyMetric:
        def set(self, value): pass
        def inc(self, value=1): pass
        def observe(self, value): pass
        def time(self): return DummyContext()
    
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    _loss_gauge = _grad_gauge = _weight_gauge = _neurons_gauge = DummyMetric()
    _consciousness_gauge = _act_entropy_gauge = _act_choice_gauge = _act_weight_gauge = DummyMetric()
    _births_counter = _deaths_counter = _cycles_counter = _adaptations_counter = DummyMetric()
    _loss_histogram = _grad_histogram = _weight_histogram = _consciousness_histogram = DummyMetric()
    _training_summary = _evaluation_summary = DummyMetric()

_server_started = False

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_metrics(port: int = 9091):
    """Inicia servidor de métricas Prometheus"""
    global _server_started
    
    if not PROMETHEUS_AVAILABLE:
        print(f"⚠️ Prometheus não disponível, métricas mock em porta {port}")
        return
    
    if _server_started:
        return
    
    try:
        start_http_server(port)
        _server_started = True
        print(f"✅ Servidor de métricas Prometheus iniciado em :{port}")
        print(f"   URL: http://localhost:{port}/metrics")
    except OSError as e:
        print(f"⚠️ Erro ao iniciar servidor de métricas: {e}")

def observe_step(loss: float, grad_norm: float, weight_norm: float,
                act_entropy: float, act_choice: str, act_weight: float):
    """Observa métricas de um step de treino"""
    
    # Atualizar gauges
    _loss_gauge.set(loss)
    _grad_gauge.set(grad_norm)
    _weight_gauge.set(weight_norm)
    _act_entropy_gauge.set(act_entropy)
    _act_weight_gauge.set(act_weight)
    
    # Mapear ativação dominante para índice
    activation_mapping = {"relu": 0, "tanh": 1, "gelu": 2, "silu": 3}
    act_idx = activation_mapping.get(act_choice.lower(), -1)
    _act_choice_gauge.set(act_idx)
    
    # Atualizar histogramas
    _loss_histogram.observe(loss)
    _grad_histogram.observe(grad_norm)
    _weight_histogram.observe(weight_norm)

def observe_cycle(neurons_count: int, consciousness_level: float, 
                 training_time: float, evaluation_time: float):
    """Observa métricas de um ciclo completo"""
    _neurons_gauge.set(neurons_count)
    _consciousness_gauge.set(consciousness_level)
    _consciousness_histogram.observe(consciousness_level)
    _cycles_counter.inc()
    
    # Não usar decorador @time para compatibilidade
    _training_summary.observe(training_time)
    _evaluation_summary.observe(evaluation_time)

def observe_birth():
    """Registra nascimento de neurônio"""
    _births_counter.inc()

def observe_death():
    """Registra morte de neurônio"""
    _deaths_counter.inc()

def observe_adaptation():
    """Registra adaptação bem-sucedida"""
    _adaptations_counter.inc()

def get_metrics_summary() -> Dict[str, Any]:
    """Retorna resumo das métricas atuais"""
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus não disponível"}
    
    try:
        import urllib.request
        url = f"http://localhost:{9091}/metrics"  # Porta padrão
        
        with urllib.request.urlopen(url, timeout=5) as response:
            content = response.read().decode('utf-8')
        
        # Parse básico das métricas
        metrics = {}
        for line in content.split('\n'):
            if line.startswith('ia3_') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    try:
                        value = float(parts[1])
                        metrics[key] = value
                    except ValueError:
                        pass
        
        return metrics
    
    except Exception as e:
        return {"error": f"Erro ao buscar métricas: {e}"}

if __name__ == "__main__":
    import sys
    
    # Teste básico das métricas
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9091
        
        print(f"🧪 Testando métricas IA³ na porta {port}...")
        ensure_metrics(port)
        
        # Simular algumas observações
        for i in range(10):
            observe_step(
                loss=0.1 - i*0.01,
                grad_norm=0.5 + i*0.1,
                weight_norm=1.0 + i*0.2,
                act_entropy=1.4 - i*0.1,
                act_choice="gelu" if i % 2 == 0 else "relu",
                act_weight=0.6 + i*0.03
            )
            
            observe_cycle(
                neurons_count=4 + i,
                consciousness_level=0.3 + i*0.05,
                training_time=0.5 + i*0.1,
                evaluation_time=0.2 + i*0.02
            )
            
            if i % 3 == 0:
                observe_birth()
            if i % 5 == 0:
                observe_death()
                
            time.sleep(0.1)
        
        print(f"✅ Teste concluído. Métricas disponíveis em http://localhost:{port}/metrics")
        
        # Mostrar resumo
        summary = get_metrics_summary()
        if "error" not in summary:
            print(f"\n📊 Resumo das métricas:")
            for key, value in sorted(summary.items()):
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
    else:
        print("Uso: python ia3_metrics.py --test [port]")