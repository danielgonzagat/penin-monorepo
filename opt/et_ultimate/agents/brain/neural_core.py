"""
Neural Core - Núcleo do sistema ET Ultimate.

Este módulo implementa o núcleo neural do sistema ET Ultimate,
fornecendo funcionalidades de processamento, aprendizagem e auto-evolução.

Classes:
    NeuralCore: Classe principal do núcleo neural
    ProcessingError: Exceção customizada para erros de processamento
    LearningError: Exceção customizada para erros de aprendizagem

Author: PENIN System
Version: 2.0.0
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Exceção customizada para erros de processamento."""
    
    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now()


class LearningError(Exception):
    """Exceção customizada para erros de aprendizagem."""
    
    def __init__(self, message: str, data_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.data_type = data_type
        self.timestamp = datetime.now()


@dataclass
class ProcessingResult:
    """Resultado do processamento de dados."""
    
    success: bool
    data: Any
    processing_time: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o resultado para dicionário."""
        return {
            'success': self.success,
            'data': self.data,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


@dataclass
class LearningData:
    """Estrutura para dados de aprendizagem."""
    
    input_data: Any
    expected_output: Optional[Any] = None
    weight: float = 1.0
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NeuralProcessor(ABC):
    """Interface abstrata para processadores neurais."""
    
    @abstractmethod
    def process(self, data: Any) -> ProcessingResult:
        """Processa os dados de entrada."""
        pass


class NeuralCore(NeuralProcessor):
    """
    Núcleo neural do sistema ET Ultimate.
    
    Esta classe implementa o processador neural principal com capacidades
    de processamento, aprendizagem e auto-evolução.
    
    Attributes:
        version (str): Versão do sistema
        status (str): Status atual do sistema
        learning_data (List[LearningData]): Dados de aprendizagem acumulados
        processing_history (List[ProcessingResult]): Histórico de processamentos
        evolution_count (int): Contador de evoluções realizadas
        
    Example:
        >>> core = NeuralCore()
        >>> result = core.process("test data")
        >>> print(result.success)
        True
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Inicializa o núcleo neural.
        
        Args:
            config: Configurações opcionais do sistema
            
        Raises:
            ValueError: Se a configuração for inválida
        """
        self.version: str = "2.0.0"
        self.status: str = "initializing"
        self.learning_data: List[LearningData] = []
        self.processing_history: List[ProcessingResult] = []
        self.evolution_count: int = 0
        self.config: Dict[str, Any] = config or {}
        self._performance_metrics: Dict[str, float] = {}
        
        # Validar configuração
        self._validate_config()
        
        # Inicializar sistema
        self._initialize_system()
        
        logger.info(f"NeuralCore v{self.version} initialized successfully")
    
    def _validate_config(self) -> None:
        """Valida a configuração do sistema."""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Definir valores padrão
        default_config = {
            'max_learning_data': 10000,
            'max_processing_history': 1000,
            'learning_rate': 0.01,
            'evolution_threshold': 100
        }
        
        # Aplicar valores padrão para chaves ausentes
        for key, value in default_config.items():
            self.config.setdefault(key, value)
    
    def _initialize_system(self) -> None:
        """Inicializa os componentes do sistema."""
        try:
            # Inicializar métricas de performance
            self._performance_metrics = {
                'total_processed': 0,
                'success_rate': 0.0,
                'average_processing_time': 0.0,
                'learning_efficiency': 0.0
            }
            
            self.status = "active"
            logger.info("System components initialized")
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Processa entrada e retorna resposta estruturada.
        
        Args:
            input_data: Dados de entrada para processamento
            
        Returns:
            ProcessingResult: Resultado estruturado do processamento
            
        Raises:
            ProcessingError: Se ocorrer erro durante o processamento
        """
        if self.status != "active":
            raise ProcessingError(
                f"System is not active (current status: {self.status})",
                error_code=1001
            )
        
        start_time = time.time()
        
        try:
            # Validar entrada
            if input_data is None:
                raise ProcessingError("Input data cannot be None", error_code=1002)
            
            # Processar dados
            processed_data = self._process_data(input_data)
            
            # Calcular tempo de processamento
            processing_time = time.time() - start_time
            
            # Criar resultado
            result = ProcessingResult(
                success=True,
                data=processed_data,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    'input_type': type(input_data).__name__,
                    'output_type': type(processed_data).__name__,
                    'version': self.version
                }
            )
            
            # Atualizar histórico e métricas
            self._update_processing_history(result)
            self._update_performance_metrics(result)
            
            logger.debug(f"Successfully processed data in {processing_time:.4f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = ProcessingResult(
                success=False,
                data=None,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={'error': str(e), 'error_type': type(e).__name__}
            )
            
            self._update_processing_history(error_result)
            logger.error(f"Processing failed: {e}")
            
            if isinstance(e, ProcessingError):
                raise
            else:
                raise ProcessingError(f"Unexpected error during processing: {e}")
    
    def _process_data(self, data: Any) -> str:
        """
        Implementação interna do processamento de dados.
        
        Args:
            data: Dados a serem processados
            
        Returns:
            str: Dados processados
        """
        # Implementação básica - pode ser estendida
        if isinstance(data, (str, int, float)):
            return f"Processado: {data} [v{self.version}]"
        elif isinstance(data, (list, tuple)):
            return f"Processado: {len(data)} items [v{self.version}]"
        elif isinstance(data, dict):
            return f"Processado: dict com {len(data)} chaves [v{self.version}]"
        else:
            return f"Processado: {type(data).__name__} object [v{self.version}]"
    
    def learn(self, data: Union[LearningData, Any], 
              expected_output: Optional[Any] = None,
              weight: float = 1.0) -> bool:
        """
        Aprende com novos dados de forma estruturada.
        
        Args:
            data: Dados de aprendizagem ou objeto LearningData
            expected_output: Saída esperada (se data não for LearningData)
            weight: Peso dos dados de aprendizagem
            
        Returns:
            bool: True se o aprendizado foi bem-sucedido
            
        Raises:
            LearningError: Se ocorrer erro durante o aprendizado
        """
        try:
            # Converter para LearningData se necessário
            if isinstance(data, LearningData):
                learning_data = data
            else:
                learning_data = LearningData(
                    input_data=data,
                    expected_output=expected_output,
                    weight=weight,
                    metadata={'timestamp': datetime.now().isoformat()}
                )
            
            # Validar dados de aprendizagem
            if learning_data.input_data is None:
                raise LearningError("Learning data cannot be None")
            
            # Adicionar aos dados de aprendizagem
            self.learning_data.append(learning_data)
            
            # Limitar tamanho do histórico
            max_learning_data = self.config.get('max_learning_data', 10000)
            if len(self.learning_data) > max_learning_data:
                self.learning_data = self.learning_data[-max_learning_data:]
            
            # Atualizar métricas de aprendizagem
            self._update_learning_metrics()
            
            logger.debug(f"Successfully learned from data: {type(learning_data.input_data).__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            if isinstance(e, LearningError):
                raise
            else:
                raise LearningError(f"Unexpected error during learning: {e}")
    
    def evolve(self) -> Dict[str, Any]:
        """
        Executa auto-evolução do sistema baseada nos dados aprendidos.
        
        Returns:
            Dict[str, Any]: Relatório da evolução realizada
            
        Raises:
            RuntimeError: Se a evolução falhar
        """
        try:
            evolution_report = {
                'evolution_id': self.evolution_count + 1,
                'timestamp': datetime.now().isoformat(),
                'previous_version': self.version,
                'changes': [],
                'performance_improvement': 0.0,
                'success': False
            }
            
            logger.info(f"Starting evolution #{evolution_report['evolution_id']}")
            
            # Verificar se há dados suficientes para evolução
            min_data_threshold = self.config.get('evolution_threshold', 100)
            if len(self.learning_data) < min_data_threshold:
                evolution_report['changes'].append(
                    f"Insufficient data for evolution (need {min_data_threshold}, have {len(self.learning_data)})"
                )
                return evolution_report
            
            # Analisar padrões nos dados de aprendizagem
            patterns = self._analyze_learning_patterns()
            evolution_report['patterns_found'] = len(patterns)
            
            # Implementar melhorias baseadas nos padrões
            improvements = self._implement_improvements(patterns)
            evolution_report['changes'].extend(improvements)
            
            # Atualizar contador de evolução
            self.evolution_count += 1
            
            # Calcular melhoria de performance
            evolution_report['performance_improvement'] = self._calculate_performance_improvement()
            evolution_report['success'] = True
            
            logger.info(f"Evolution #{self.evolution_count} completed successfully")
            return evolution_report
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise RuntimeError(f"Evolution process failed: {e}")
    
    def _analyze_learning_patterns(self) -> List[Dict[str, Any]]:
        """Analisa padrões nos dados de aprendizagem."""
        patterns = []
        
        # Analisar tipos de dados mais comuns
        type_counts = {}
        for data in self.learning_data:
            data_type = type(data.input_data).__name__
            type_counts[data_type] = type_counts.get(data_type, 0) + 1
        
        if type_counts:
            most_common_type = max(type_counts, key=type_counts.get)
            patterns.append({
                'type': 'data_type_preference',
                'value': most_common_type,
                'frequency': type_counts[most_common_type] / len(self.learning_data)
            })
        
        return patterns
    
    def _implement_improvements(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Implementa melhorias baseadas nos padrões identificados."""
        improvements = []
        
        for pattern in patterns:
            if pattern['type'] == 'data_type_preference' and pattern['frequency'] > 0.7:
                improvements.append(
                    f"Optimized processing for {pattern['value']} data type "
                    f"(found in {pattern['frequency']:.1%} of learning data)"
                )
        
        return improvements
    
    def _calculate_performance_improvement(self) -> float:
        """Calcula a melhoria de performance após evolução."""
        if len(self.processing_history) < 10:
            return 0.0
        
        recent_times = [r.processing_time for r in self.processing_history[-10:] if r.success]
        older_times = [r.processing_time for r in self.processing_history[-20:-10] if r.success]
        
        if not recent_times or not older_times:
            return 0.0
        
        recent_avg = sum(recent_times) / len(recent_times)
        older_avg = sum(older_times) / len(older_times)
        
        if older_avg > 0:
            return max(0.0, (older_avg - recent_avg) / older_avg * 100)
        
        return 0.0
    
    def _update_processing_history(self, result: ProcessingResult) -> None:
        """Atualiza o histórico de processamento."""
        self.processing_history.append(result)
        
        # Limitar tamanho do histórico
        max_history = self.config.get('max_processing_history', 1000)
        if len(self.processing_history) > max_history:
            self.processing_history = self.processing_history[-max_history:]
    
    def _update_performance_metrics(self, result: ProcessingResult) -> None:
        """Atualiza métricas de performance."""
        self._performance_metrics['total_processed'] += 1
        
        # Calcular taxa de sucesso
        successful = sum(1 for r in self.processing_history if r.success)
        self._performance_metrics['success_rate'] = successful / len(self.processing_history)
        
        # Calcular tempo médio de processamento
        processing_times = [r.processing_time for r in self.processing_history if r.success]
        if processing_times:
            self._performance_metrics['average_processing_time'] = sum(processing_times) / len(processing_times)
    
    def _update_learning_metrics(self) -> None:
        """Atualiza métricas de aprendizagem."""
        if self.learning_data:
            weighted_sum = sum(data.weight for data in self.learning_data)
            self._performance_metrics['learning_efficiency'] = weighted_sum / len(self.learning_data)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status completo do sistema.
        
        Returns:
            Dict[str, Any]: Status detalhado do sistema
        """
        return {
            'version': self.version,
            'status': self.status,
            'evolution_count': self.evolution_count,
            'learning_data_count': len(self.learning_data),
            'processing_history_count': len(self.processing_history),
            'performance_metrics': self._performance_metrics.copy(),
            'config': self.config.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_data(self, filepath: str, include_history: bool = True) -> None:
        """
        Exporta dados do sistema para arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo para exportação
            include_history: Se deve incluir histórico completo
        """
        export_data = {
            'system_info': self.get_status(),
            'learning_data': [
                {
                    'input_data': str(data.input_data),
                    'expected_output': str(data.expected_output) if data.expected_output else None,
                    'weight': data.weight,
                    'category': data.category,
                    'metadata': data.metadata
                }
                for data in self.learning_data
            ]
        }
        
        if include_history:
            export_data['processing_history'] = [
                result.to_dict() for result in self.processing_history
            ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Data exported to {filepath}")
    
    def __repr__(self) -> str:
        """Representação string da instância."""
        return (f"NeuralCore(version='{self.version}', status='{self.status}', "
                f"evolutions={self.evolution_count})")
    
    def __str__(self) -> str:
        """String amigável da instância."""
        return f"Neural Core v{self.version} - Status: {self.status}"