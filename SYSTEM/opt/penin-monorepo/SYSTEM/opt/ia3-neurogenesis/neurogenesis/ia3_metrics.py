try:
    from prometheus_client import start_http_server, Counter, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class Metrics:
    def __init__(self, port=9092):
        self.port = port
        self._started = False
        self._gauges = {}
        self._counters = {}
        self.available = PROMETHEUS_AVAILABLE
    
    def start(self):
        if not self.available:
            print(f"⚠️ Prometheus não disponível, métricas desabilitadas")
            return
            
        if not self._started:
            try:
                start_http_server(self.port)
                self._started = True
                print(f"✅ Métricas Prometheus ativas em :{self.port}")
            except OSError as e:
                print(f"⚠️ Erro ao iniciar servidor de métricas: {e}")
    
    def gauge(self, name):
        if not self.available:
            return DummyMetric()
        if name not in self._gauges:
            self._gauges[name] = Gauge(f"ia3_{name}", f"IA3 gauge {name}")
        return self._gauges[name]
    
    def counter(self, name):
        if not self.available:
            return DummyMetric()
        if name not in self._counters:
            self._counters[name] = Counter(f"ia3_{name}", f"IA3 counter {name}")
        return self._counters[name]

class DummyMetric:
    """Placeholder for when prometheus is not available"""
    def set(self, value): pass
    def inc(self, value=1): pass