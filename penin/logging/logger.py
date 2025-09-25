"""
Advanced Logging and Monitoring System
Sistema de logging estruturado com métricas, alertas e observabilidade
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
from functools import wraps
import traceback

# Third-party imports (will be available after pip install)
try:
    from loguru import logger as loguru_logger
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import structlog
    ADVANCED_LOGGING = True
except ImportError:
    ADVANCED_LOGGING = False
    loguru_logger = None

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class LogEntry:
    """Entrada estruturada de log"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    extra_data: Dict[str, Any]
    stack_trace: Optional[str] = None

@dataclass
class MetricEntry:
    """Entrada de métrica"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str]
    timestamp: str
    description: str

class PENINLogger:
    """
    Sistema de logging avançado com:
    - Logging estruturado
    - Métricas e monitoramento
    - Alertas automáticos
    - Correlação de eventos
    - Exportação para múltiplos backends
    """
    
    def __init__(self, name: str = "penin", config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.metrics = {}
        self.alert_handlers = []
        self.log_handlers = []
        self.correlation_id = None
        
        # Initialize logging system
        self._setup_logging()
        self._setup_metrics()
        
        # Start metrics server if enabled
        if self.config.get('metrics_enabled', True):
            self._start_metrics_server()
        
        self.logger.info(f"PENIN Logger initialized - {name}")
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_level = self.config.get('log_level', 'INFO')
        log_format = self.config.get('log_format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure standard Python logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if configured
        log_file = self.config.get('log_file')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Setup advanced logging if available
        if ADVANCED_LOGGING:
            self._setup_advanced_logging()
    
    def _setup_advanced_logging(self):
        """Configura logging avançado com loguru e structlog"""
        if loguru_logger:
            # Configure loguru
            loguru_logger.remove()  # Remove default handler
            
            # Add console handler
            loguru_logger.add(
                sys.stdout,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
                level=self.config.get('log_level', 'INFO'),
                colorize=True
            )
            
            # Add file handler if configured
            log_file = self.config.get('log_file')
            if log_file:
                loguru_logger.add(
                    log_file,
                    rotation="10 MB",
                    retention="30 days",
                    compression="zip",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message} | {extra}",
                    serialize=True  # JSON format
                )
        
        # Configure structlog if available
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            self.struct_logger = structlog.get_logger(self.name)
        except:
            self.struct_logger = None
    
    def _setup_metrics(self):
        """Configura sistema de métricas"""
        if not ADVANCED_LOGGING:
            return
        
        try:
            # Métricas básicas do sistema
            self.metrics['requests_total'] = Counter(
                'penin_requests_total',
                'Total requests processed',
                ['method', 'endpoint', 'status']
            )
            
            self.metrics['request_duration'] = Histogram(
                'penin_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            )
            
            self.metrics['neural_operations'] = Counter(
                'penin_neural_operations_total',
                'Neural operations processed',
                ['module', 'operation']
            )
            
            self.metrics['system_health'] = Gauge(
                'penin_system_health',
                'System health score (0-1)',
                ['component']
            )
            
            self.metrics['memory_usage'] = Gauge(
                'penin_memory_usage_bytes',
                'Memory usage in bytes',
                ['component']
            )
            
            self.metrics['evolution_count'] = Counter(
                'penin_evolution_total',
                'Total evolution events',
                ['component']
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to setup metrics: {e}")
    
    def _start_metrics_server(self):
        """Inicia servidor de métricas Prometheus"""
        if not ADVANCED_LOGGING:
            return
        
        try:
            port = self.config.get('metrics_port', 9090)
            start_http_server(port)
            self.logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            self.logger.warning(f"Failed to start metrics server: {e}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['stack_trace'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['stack_trace'] = traceback.format_exc()
        self._log(LogLevel.CRITICAL, message, **kwargs)
        
        # Trigger alerts for critical messages
        self._trigger_alerts(LogLevel.CRITICAL, message, kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method"""
        # Add correlation ID if available
        if self.correlation_id:
            kwargs['correlation_id'] = self.correlation_id
        
        # Add system context
        kwargs.update({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'thread_id': threading.current_thread().name,
            'process_id': os.getpid()
        })
        
        # Log with standard logger
        log_method = getattr(self.logger, level.value.lower())
        log_method(f"{message} | {json.dumps(kwargs)}")
        
        # Log with advanced loggers if available
        if self.struct_logger:
            struct_log_method = getattr(self.struct_logger, level.value.lower())
            struct_log_method(message, **kwargs)
        
        if loguru_logger:
            loguru_log_method = getattr(loguru_logger, level.value.lower())
            loguru_log_method(f"{message} | {kwargs}")
        
        # Create structured log entry
        frame = sys._getframe(2)
        log_entry = LogEntry(
            timestamp=kwargs['timestamp'],
            level=level.value,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=kwargs['thread_id'],
            process_id=kwargs['process_id'],
            extra_data=kwargs,
            stack_trace=kwargs.get('stack_trace')
        )
        
        # Send to custom handlers
        for handler in self.log_handlers:
            try:
                handler(log_entry)
            except Exception as e:
                self.logger.warning(f"Log handler failed: {e}")
    
    def metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
              labels: Optional[Dict[str, str]] = None, description: str = ""):
        """Record a metric"""
        labels = labels or {}
        
        # Record with Prometheus if available
        if name in self.metrics:
            try:
                metric = self.metrics[name]
                if metric_type == MetricType.COUNTER:
                    metric.labels(**labels).inc(value)
                elif metric_type == MetricType.GAUGE:
                    metric.labels(**labels).set(value)
                elif metric_type == MetricType.HISTOGRAM:
                    metric.labels(**labels).observe(value)
            except Exception as e:
                self.logger.warning(f"Failed to record metric {name}: {e}")
        
        # Create metric entry
        metric_entry = MetricEntry(
            name=name,
            type=metric_type,
            value=value,
            labels=labels,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description
        )
        
        # Log metric
        self.debug(f"Metric recorded: {name}={value}", metric=asdict(metric_entry))
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, labels or {})
    
    def correlation(self, correlation_id: str):
        """Context manager for correlation tracking"""
        return CorrelationContext(self, correlation_id)
    
    def add_alert_handler(self, handler: Callable[[LogLevel, str, Dict], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def add_log_handler(self, handler: Callable[[LogEntry], None]):
        """Add custom log handler"""
        self.log_handlers.append(handler)
    
    def _trigger_alerts(self, level: LogLevel, message: str, context: Dict):
        """Trigger alert handlers"""
        for handler in self.alert_handlers:
            try:
                handler(level, message, context)
            except Exception as e:
                self.logger.warning(f"Alert handler failed: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recorded metrics"""
        return {
            'metrics_count': len(self.metrics),
            'handlers_count': len(self.log_handlers),
            'alert_handlers_count': len(self.alert_handlers),
            'correlation_id': self.correlation_id
        }

class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, logger: PENINLogger, name: str, labels: Dict[str, str]):
        self.logger = logger
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.metric(
                f"{self.name}_duration",
                duration,
                MetricType.HISTOGRAM,
                self.labels,
                f"Duration of {self.name} operation"
            )

class CorrelationContext:
    """Context manager for correlation ID tracking"""
    
    def __init__(self, logger: PENINLogger, correlation_id: str):
        self.logger = logger
        self.correlation_id = correlation_id
        self.previous_correlation_id = None
    
    def __enter__(self):
        self.previous_correlation_id = self.logger.correlation_id
        self.logger.correlation_id = self.correlation_id
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.correlation_id = self.previous_correlation_id

# Decorators for automatic logging
def log_function_calls(logger: PENINLogger, level: LogLevel = LogLevel.DEBUG):
    """Decorator to log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            with logger.timer(f"function_{func_name}"):
                logger._log(level, f"Calling function {func_name}", 
                           args_count=len(args), kwargs_keys=list(kwargs.keys()))
                
                try:
                    result = func(*args, **kwargs)
                    logger._log(level, f"Function {func_name} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Function {func_name} failed", exception=e)
                    raise
        
        return wrapper
    return decorator

def log_neural_operations(logger: PENINLogger):
    """Decorator for neural operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = func.__name__
            
            # Record neural operation metric
            logger.metric('neural_operations', 1, MetricType.COUNTER, 
                         {'operation': operation_name})
            
            with logger.timer(f"neural_{operation_name}"):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global logger instance
_global_logger = None

def get_logger(name: str = "penin", config: Optional[Dict] = None) -> PENINLogger:
    """Get or create logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PENINLogger(name, config)
    return _global_logger

def setup_logging(config: Dict[str, Any]) -> PENINLogger:
    """Setup logging system with configuration"""
    return PENINLogger("penin", config)

# Alert handlers
def email_alert_handler(level: LogLevel, message: str, context: Dict):
    """Email alert handler (placeholder)"""
    if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
        print(f"EMAIL ALERT: {level.value} - {message}")

def slack_alert_handler(level: LogLevel, message: str, context: Dict):
    """Slack alert handler (placeholder)"""
    if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
        print(f"SLACK ALERT: {level.value} - {message}")

# Example usage and testing
if __name__ == "__main__":
    # Test logging system
    config = {
        'log_level': 'DEBUG',
        'log_file': '/workspace/logs/penin.log',
        'metrics_enabled': True,
        'metrics_port': 9090
    }
    
    logger = setup_logging(config)
    
    # Add alert handlers
    logger.add_alert_handler(email_alert_handler)
    logger.add_alert_handler(slack_alert_handler)
    
    # Test logging
    logger.info("System starting up", component="main")
    logger.debug("Debug information", data={"key": "value"})
    logger.warning("Warning message", alert=True)
    
    # Test metrics
    logger.metric("system_health", 0.95, MetricType.GAUGE, {"component": "neural_core"})
    logger.metric("requests_total", 1, MetricType.COUNTER, {"endpoint": "/api/test"})
    
    # Test timer
    with logger.timer("test_operation"):
        time.sleep(0.1)
    
    # Test correlation
    with logger.correlation("test-correlation-123"):
        logger.info("Operation within correlation context")
    
    # Test error logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Test error occurred", exception=e)
    
    print("Metrics Summary:", logger.get_metrics_summary())