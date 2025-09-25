"""
Comprehensive test suite for the Neural Core module.

This module provides unit tests for all functionality of the NeuralCore class,
including performance testing, error handling, and edge cases.
"""

import pytest
import time
import logging
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from neural_core import (
    NeuralCore,
    SystemStatus,
    ProcessingResult,
    NeuralInterface
)


class TestNeuralCore:
    """Test suite for the NeuralCore class."""
    
    def setup_method(self) -> None:
        """Setup method called before each test."""
        self.neural_core = NeuralCore()
    
    def test_initialization_default(self) -> None:
        """Test default initialization of NeuralCore."""
        core = NeuralCore()
        assert core.version == "1.0.0"
        assert core.status == SystemStatus.ACTIVE
        assert isinstance(core.learning_data, list)
        assert len(core.learning_data) == 0
        assert isinstance(core.processing_cache, dict)
        assert len(core.processing_cache) == 0
    
    def test_initialization_custom(self) -> None:
        """Test custom initialization parameters."""
        core = NeuralCore(version="2.0.0", cache_size=500)
        assert core.version == "2.0.0"
        assert core._cache_size == 500
    
    def test_initialization_invalid_cache_size(self) -> None:
        """Test initialization with invalid cache size."""
        with pytest.raises(ValueError, match="Cache size must be non-negative"):
            NeuralCore(cache_size=-1)
    
    def test_process_basic(self) -> None:
        """Test basic data processing functionality."""
        result = self.neural_core.process("test_data")
        
        assert isinstance(result, ProcessingResult)
        assert result.result == "Processado: test_data"
        assert result.status == SystemStatus.ACTIVE
        assert isinstance(result.processing_time, float)
        assert result.processing_time >= 0
        assert "cached" in result.metadata
        assert result.metadata["cached"] is False
    
    def test_process_caching(self) -> None:
        """Test caching functionality in processing."""
        # First call - should not be cached
        result1 = self.neural_core.process("cache_test")
        assert result1.metadata["cached"] is False
        
        # Second call - should be cached
        result2 = self.neural_core.process("cache_test")
        assert result2.metadata["cached"] is True
        assert result2.result == result1.result
    
    def test_process_none_input(self) -> None:
        """Test processing with None input."""
        with pytest.raises(TypeError, match="Input data cannot be None"):
            self.neural_core.process(None)
    
    def test_process_error_state(self) -> None:
        """Test processing when system is in error state."""
        self.neural_core.status = SystemStatus.ERROR
        
        with pytest.raises(RuntimeError, match="Cannot process data: system is in error state"):
            self.neural_core.process("test")
    
    def test_learn_basic(self) -> None:
        """Test basic learning functionality."""
        data = ["item1", "item2", "item3"]
        result = self.neural_core.learn(data)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 3
        assert self.neural_core.learning_data == data
    
    def test_learn_dict_data(self) -> None:
        """Test learning with dictionary data."""
        data = {"key1": "value1", "key2": "value2"}
        result = self.neural_core.learn(data)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 2
        assert "value1" in self.neural_core.learning_data
        assert "value2" in self.neural_core.learning_data
    
    def test_learn_single_item(self) -> None:
        """Test learning with single item."""
        data = "single_item"
        result = self.neural_core.learn(data)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 1
        assert self.neural_core.learning_data[0] == "single_item"
    
    def test_learn_empty_data(self) -> None:
        """Test learning with empty data."""
        with pytest.raises(ValueError, match="Learning data cannot be empty"):
            self.neural_core.learn([])
        
        with pytest.raises(ValueError, match="Learning data cannot be empty"):
            self.neural_core.learn({})
    
    def test_learn_memory_limit(self) -> None:
        """Test learning data memory limit."""
        # Create large dataset to trigger memory limit
        large_data = [f"item_{i}" for i in range(15000)]
        result = self.neural_core.learn(large_data)
        
        assert result is True
        assert len(self.neural_core.learning_data) == 10000  # Should be truncated
    
    def test_evolve_basic(self) -> None:
        """Test basic evolution functionality."""
        result = self.neural_core.evolve()
        
        assert isinstance(result, dict)
        assert "evolution_count" in result
        assert result["evolution_count"] == 1
        assert "learning_data_size" in result
        assert "cache_efficiency" in result
        assert "optimizations_applied" in result
        assert "evolution_time" in result
        assert isinstance(result["evolution_time"], float)
    
    def test_evolve_error_state(self) -> None:
        """Test evolution when system is in error state."""
        self.neural_core.status = SystemStatus.ERROR
        
        with pytest.raises(RuntimeError, match="Cannot evolve: system is in error state"):
            self.neural_core.evolve()
    
    def test_evolve_cache_cleanup(self) -> None:
        """Test cache cleanup during evolution."""
        # Fill cache beyond 80% capacity
        core = NeuralCore(cache_size=100)
        for i in range(85):
            core.process(f"data_{i}")
        
        result = core.evolve()
        assert "cache_cleanup" in result["optimizations_applied"]
    
    def test_evolve_version_increment(self) -> None:
        """Test version increment during evolution."""
        # Trigger multiple evolutions to test version increment
        for _ in range(10):
            self.neural_core.evolve()
        
        result = self.neural_core.evolve()
        assert "version_increment" in result["optimizations_applied"]
        assert self.neural_core.version == "1.0.1"
    
    def test_cache_efficiency(self) -> None:
        """Test cache efficiency calculation."""
        # Empty cache
        assert self.neural_core._calculate_cache_efficiency() == 0.0
        
        # Partially filled cache
        self.neural_core.process("test1")
        self.neural_core.process("test2")
        efficiency = self.neural_core._calculate_cache_efficiency()
        assert 0.0 < efficiency <= 1.0
    
    def test_get_system_info(self) -> None:
        """Test system information retrieval."""
        info = self.neural_core.get_system_info()
        
        assert isinstance(info, dict)
        assert "version" in info
        assert "status" in info
        assert "evolution_count" in info
        assert "learning_data_size" in info
        assert "cache_size" in info
        assert "cache_capacity" in info
        assert "cache_efficiency" in info
        
        assert info["version"] == "1.0.0"
        assert info["status"] == "active"
    
    def test_reset(self) -> None:
        """Test system reset functionality."""
        # Add some data and perform operations
        self.neural_core.learn(["test_data"])
        self.neural_core.process("test_input")
        self.neural_core.evolve()
        
        # Verify data exists
        assert len(self.neural_core.learning_data) > 0
        assert len(self.neural_core.processing_cache) > 0
        assert self.neural_core._evolution_count > 0
        
        # Reset and verify
        self.neural_core.reset()
        assert len(self.neural_core.learning_data) == 0
        assert len(self.neural_core.processing_cache) == 0
        assert self.neural_core._evolution_count == 0
        assert self.neural_core.status == SystemStatus.ACTIVE
    
    def test_string_representations(self) -> None:
        """Test string representation methods."""
        str_repr = str(self.neural_core)
        assert "NeuralCore" in str_repr
        assert "1.0.0" in str_repr
        assert "active" in str_repr
        
        repr_str = repr(self.neural_core)
        assert "NeuralCore" in repr_str
        assert "version='1.0.0'" in repr_str
        assert "status=SystemStatus.ACTIVE" in repr_str
    
    def test_logging_setup(self) -> None:
        """Test logging configuration."""
        assert hasattr(self.neural_core, 'logger')
        assert isinstance(self.neural_core.logger, logging.Logger)
    
    def test_performance_timing(self) -> None:
        """Test performance timing accuracy."""
        start_time = time.perf_counter()
        result = self.neural_core.process("performance_test")
        end_time = time.perf_counter()
        
        # Processing time should be reasonable
        assert result.processing_time <= (end_time - start_time) + 0.001  # Small tolerance
        assert result.processing_time >= 0
    
    def test_cache_overflow_handling(self) -> None:
        """Test cache behavior when exceeding capacity."""
        core = NeuralCore(cache_size=3)
        
        # Fill cache to capacity
        core.process("item1")
        core.process("item2")
        core.process("item3")
        assert len(core.processing_cache) == 3
        
        # Add one more item - should trigger cache cleanup
        core.process("item4")
        assert len(core.processing_cache) == 3  # Should maintain capacity
    
    def test_status_transitions(self) -> None:
        """Test status transitions during operations."""
        initial_status = self.neural_core.status
        
        # Test learning status transition
        self.neural_core.learn(["test"])
        assert self.neural_core.status == initial_status  # Should return to original
        
        # Test evolution status transition
        self.neural_core.evolve()
        assert self.neural_core.status == initial_status  # Should return to original


class TestSystemStatus:
    """Test suite for SystemStatus enum."""
    
    def test_status_values(self) -> None:
        """Test SystemStatus enum values."""
        assert SystemStatus.ACTIVE.value == "active"
        assert SystemStatus.INACTIVE.value == "inactive"
        assert SystemStatus.LEARNING.value == "learning"
        assert SystemStatus.EVOLVING.value == "evolving"
        assert SystemStatus.ERROR.value == "error"


class TestProcessingResult:
    """Test suite for ProcessingResult dataclass."""
    
    def test_processing_result_creation(self) -> None:
        """Test ProcessingResult dataclass creation."""
        result = ProcessingResult(
            result="test_result",
            processing_time=0.001,
            status=SystemStatus.ACTIVE
        )
        
        assert result.result == "test_result"
        assert result.processing_time == 0.001
        assert result.status == SystemStatus.ACTIVE
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0
    
    def test_processing_result_with_metadata(self) -> None:
        """Test ProcessingResult with custom metadata."""
        metadata = {"cached": True, "source": "test"}
        result = ProcessingResult(
            result="test_result",
            processing_time=0.001,
            status=SystemStatus.ACTIVE,
            metadata=metadata
        )
        
        assert result.metadata == metadata


class TestNeuralInterface:
    """Test suite for NeuralInterface abstract class."""
    
    def test_interface_implementation(self) -> None:
        """Test that NeuralCore properly implements NeuralInterface."""
        assert isinstance(NeuralCore(), NeuralInterface)
        
        # Verify abstract methods are implemented
        core = NeuralCore()
        assert hasattr(core, 'process')
        assert hasattr(core, 'learn')
        assert callable(getattr(core, 'process'))
        assert callable(getattr(core, 'learn'))


# Performance and integration tests
class TestPerformanceAndIntegration:
    """Performance and integration test suite."""
    
    def test_processing_performance(self) -> None:
        """Test processing performance with various data sizes."""
        core = NeuralCore()
        
        # Test with different data sizes
        test_data = [
            "small",
            "medium_sized_data_string_for_testing",
            "large_data_string_" * 100,
            {"complex": {"nested": {"data": "structure"}}}
        ]
        
        for data in test_data:
            start_time = time.perf_counter()
            result = core.process(data)
            end_time = time.perf_counter()
            
            # Should complete within reasonable time
            assert (end_time - start_time) < 0.1  # 100ms max
            assert result.status == SystemStatus.ACTIVE
    
    def test_concurrent_operations(self) -> None:
        """Test system behavior under concurrent operations."""
        core = NeuralCore()
        
        # Simulate concurrent operations
        results = []
        for i in range(10):
            # Mix of operations
            if i % 3 == 0:
                result = core.process(f"data_{i}")
                results.append(result)
            elif i % 3 == 1:
                success = core.learn([f"learning_data_{i}"])
                assert success is True
            else:
                evolution_result = core.evolve()
                assert "evolution_count" in evolution_result
        
        # Verify all processing results are valid
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.status in [SystemStatus.ACTIVE, SystemStatus.ERROR]
    
    def test_memory_usage_stability(self) -> None:
        """Test memory usage stability over extended operations."""
        core = NeuralCore(cache_size=100)
        
        # Perform many operations
        for i in range(1000):
            core.process(f"data_{i % 50}")  # Some repetition for cache testing
            
            if i % 10 == 0:
                core.learn([f"learning_{i}"])
            
            if i % 50 == 0:
                core.evolve()
        
        # Verify system is still stable
        info = core.get_system_info()
        assert info["status"] == "active"
        assert info["cache_size"] <= 100  # Should respect cache limit
        assert info["learning_data_size"] <= 10000  # Should respect learning limit


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])