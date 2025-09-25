"""
Testes unitários para o Neural Core.

Este módulo contém testes abrangentes para validar o funcionamento
do sistema Neural Core otimizado.

Author: PENIN System
Version: 2.0.0
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

from neural_core import (
    NeuralCore, ProcessingError, LearningError,
    ProcessingResult, LearningData, NeuralProcessor
)


class TestNeuralCore:
    """Testes para a classe NeuralCore."""
    
    def setup_method(self):
        """Configuração executada antes de cada teste."""
        self.config = {
            'max_learning_data': 100,
            'max_processing_history': 50,
            'learning_rate': 0.01,
            'evolution_threshold': 10
        }
        self.neural_core = NeuralCore(self.config)
    
    def test_initialization(self):
        """Testa a inicialização do NeuralCore."""
        assert self.neural_core.version == "2.0.0"
        assert self.neural_core.status == "active"
        assert self.neural_core.evolution_count == 0
        assert len(self.neural_core.learning_data) == 0
        assert len(self.neural_core.processing_history) == 0
    
    def test_initialization_with_invalid_config(self):
        """Testa inicialização com configuração inválida."""
        with pytest.raises(ValueError):
            NeuralCore("invalid_config")
    
    def test_process_string_data(self):
        """Testa processamento de dados string."""
        result = self.neural_core.process("test data")
        
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert "Processado: test data [v2.0.0]" in result.data
        assert result.processing_time > 0
        assert isinstance(result.timestamp, datetime)
        assert result.metadata['input_type'] == 'str'
        assert result.metadata['version'] == '2.0.0'
    
    def test_process_list_data(self):
        """Testa processamento de dados lista."""
        test_list = [1, 2, 3, 4, 5]
        result = self.neural_core.process(test_list)
        
        assert result.success is True
        assert "5 items" in result.data
        assert result.metadata['input_type'] == 'list'
    
    def test_process_dict_data(self):
        """Testa processamento de dados dicionário."""
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        result = self.neural_core.process(test_dict)
        
        assert result.success is True
        assert "3 chaves" in result.data
        assert result.metadata['input_type'] == 'dict'
    
    def test_process_none_data(self):
        """Testa processamento de dados None."""
        with pytest.raises(ProcessingError) as exc_info:
            self.neural_core.process(None)
        
        assert exc_info.value.error_code == 1002
        assert "cannot be None" in str(exc_info.value)
    
    def test_process_with_inactive_system(self):
        """Testa processamento com sistema inativo."""
        self.neural_core.status = "inactive"
        
        with pytest.raises(ProcessingError) as exc_info:
            self.neural_core.process("test")
        
        assert exc_info.value.error_code == 1001
        assert "not active" in str(exc_info.value)
    
    def test_learn_basic(self):
        """Testa aprendizagem básica."""
        result = self.neural_core.learn("input_data", "expected_output", 1.5)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 1
        
        learning_data = self.neural_core.learning_data[0]
        assert learning_data.input_data == "input_data"
        assert learning_data.expected_output == "expected_output"
        assert learning_data.weight == 1.5
    
    def test_learn_with_learning_data_object(self):
        """Testa aprendizagem com objeto LearningData."""
        learning_data = LearningData(
            input_data="test",
            expected_output="result",
            weight=2.0,
            category="test_category",
            metadata={'source': 'test'}
        )
        
        result = self.neural_core.learn(learning_data)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 1
        assert self.neural_core.learning_data[0].category == "test_category"
        assert self.neural_core.learning_data[0].metadata['source'] == 'test'
    
    def test_learn_with_none_data(self):
        """Testa aprendizagem com dados None."""
        with pytest.raises(LearningError):
            self.neural_core.learn(None)
    
    def test_learn_data_limit(self):
        """Testa limite de dados de aprendizagem."""
        # Configurar limite baixo
        self.neural_core.config['max_learning_data'] = 3
        
        # Adicionar mais dados que o limite
        for i in range(5):
            self.neural_core.learn(f"data_{i}")
        
        # Verificar que apenas os últimos dados foram mantidos
        assert len(self.neural_core.learning_data) == 3
        assert self.neural_core.learning_data[0].input_data == "data_2"
        assert self.neural_core.learning_data[-1].input_data == "data_4"
    
    def test_evolve_insufficient_data(self):
        """Testa evolução com dados insuficientes."""
        # Adicionar poucos dados
        for i in range(5):
            self.neural_core.learn(f"data_{i}")
        
        report = self.neural_core.evolve()
        
        assert report['success'] is False
        assert "Insufficient data" in report['changes'][0]
        assert report['evolution_id'] == 1
    
    def test_evolve_with_sufficient_data(self):
        """Testa evolução com dados suficientes."""
        # Adicionar dados suficientes
        for i in range(15):
            self.neural_core.learn(f"data_{i}")
        
        # Adicionar alguns processamentos para histórico
        for i in range(5):
            self.neural_core.process(f"test_{i}")
        
        report = self.neural_core.evolve()
        
        assert report['success'] is True
        assert report['evolution_id'] == 1
        assert self.neural_core.evolution_count == 1
        assert 'patterns_found' in report
    
    def test_get_status(self):
        """Testa obtenção de status do sistema."""
        # Adicionar alguns dados
        self.neural_core.learn("test_data")
        self.neural_core.process("test_input")
        
        status = self.neural_core.get_status()
        
        assert status['version'] == "2.0.0"
        assert status['status'] == "active"
        assert status['learning_data_count'] == 1
        assert status['processing_history_count'] == 1
        assert 'performance_metrics' in status
        assert 'timestamp' in status
    
    def test_export_data(self):
        """Testa exportação de dados."""
        # Adicionar alguns dados
        self.neural_core.learn("test_data", "expected")
        self.neural_core.process("test_input")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            self.neural_core.export_data(filepath, include_history=True)
            
            # Verificar se arquivo foi criado
            assert os.path.exists(filepath)
            
            # Verificar conteúdo
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'system_info' in data
            assert 'learning_data' in data
            assert 'processing_history' in data
            assert len(data['learning_data']) == 1
            assert len(data['processing_history']) == 1
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_performance_metrics_update(self):
        """Testa atualização de métricas de performance."""
        # Processar alguns dados
        for i in range(5):
            self.neural_core.process(f"test_{i}")
        
        metrics = self.neural_core._performance_metrics
        
        assert metrics['total_processed'] == 5
        assert metrics['success_rate'] == 1.0
        assert metrics['average_processing_time'] > 0
    
    def test_string_representations(self):
        """Testa representações string da classe."""
        repr_str = repr(self.neural_core)
        str_str = str(self.neural_core)
        
        assert "NeuralCore" in repr_str
        assert "2.0.0" in repr_str
        assert "active" in repr_str
        
        assert "Neural Core v2.0.0" in str_str
        assert "Status: active" in str_str


class TestProcessingResult:
    """Testes para a classe ProcessingResult."""
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        result = ProcessingResult(
            success=True,
            data="test_data",
            processing_time=0.123,
            timestamp=datetime.now(),
            metadata={'key': 'value'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['success'] is True
        assert result_dict['data'] == "test_data"
        assert result_dict['processing_time'] == 0.123
        assert 'timestamp' in result_dict
        assert result_dict['metadata']['key'] == 'value'


class TestExceptions:
    """Testes para exceções customizadas."""
    
    def test_processing_error(self):
        """Testa ProcessingError."""
        error = ProcessingError("Test error", error_code=1001)
        
        assert str(error) == "Test error"
        assert error.error_code == 1001
        assert isinstance(error.timestamp, datetime)
    
    def test_learning_error(self):
        """Testa LearningError."""
        error = LearningError("Learning failed", data_type="string")
        
        assert str(error) == "Learning failed"
        assert error.data_type == "string"
        assert isinstance(error.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])