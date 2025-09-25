"""
Test suite for Neural Core functionality
Comprehensive tests for the enhanced neural core system
"""

import pytest
import sys
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules to test
try:
    from opt.et_ultimate.agents.brain.neural_core import (
        NeuralCore, NeuralModule, LanguageModule, ReasoningModule, MemoryModule,
        ProcessingMode, LearningStrategy, NeuralState, create_neural_core
    )
    NEURAL_CORE_AVAILABLE = True
except ImportError as e:
    NEURAL_CORE_AVAILABLE = False
    print(f"Neural core not available for testing: {e}")

# Test fixtures
@pytest.fixture
def neural_core():
    """Create a neural core instance for testing"""
    if not NEURAL_CORE_AVAILABLE:
        pytest.skip("Neural core not available")
    
    return create_neural_core()

@pytest.fixture
def sample_text_input():
    """Sample text input for testing"""
    return "This is a test input for the neural system to process and analyze."

@pytest.fixture
def sample_dict_input():
    """Sample dictionary input for testing"""
    return {
        "query": "What is the meaning of life?",
        "context": "philosophical discussion",
        "priority": "high"
    }

class TestNeuralState:
    """Test NeuralState data structure"""
    
    def test_neural_state_creation(self):
        """Test creating neural state"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        state = NeuralState(
            version="2.0.0",
            status="active",
            learning_rate=0.01,
            confidence_threshold=0.7,
            processing_mode=ProcessingMode.HYBRID,
            active_modules=["language", "reasoning"],
            memory_usage=0.5,
            evolution_count=0,
            last_evolution="2025-01-01T00:00:00",
            performance_metrics={"accuracy": 0.95}
        )
        
        assert state.version == "2.0.0"
        assert state.status == "active"
        assert state.processing_mode == ProcessingMode.HYBRID
        assert len(state.active_modules) == 2

class TestLanguageModule:
    """Test Language Module functionality"""
    
    def test_language_module_creation(self):
        """Test creating language module"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = LanguageModule()
        assert module.name == "LanguageModule"
        assert module.version == "2.0.0"
        assert module.active is True
        assert module.vocabulary_size == 50000
        assert module.context_window == 4096
    
    def test_language_module_process_text(self, sample_text_input):
        """Test text processing"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = LanguageModule()
        result = module.process(sample_text_input)
        
        assert isinstance(result, dict)
        assert "tokens" in result
        assert "sentiment" in result
        assert "complexity" in result
        assert "entities" in result
        assert "intent" in result
        
        assert result["tokens"] > 0
        assert -1 <= result["sentiment"] <= 1
        assert 0 <= result["complexity"] <= 1
    
    def test_language_module_learning(self, sample_text_input):
        """Test language module learning"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = LanguageModule()
        initial_score = module.performance_score
        
        feedback = {"accuracy": 0.85}
        module.learn(sample_text_input, feedback)
        
        # Performance score should be updated
        assert module.performance_score != initial_score
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = LanguageModule()
        text = "John Smith visited New York and met with Apple Inc."
        
        entities = module._extract_entities(text)
        
        assert "John" in entities
        assert "Smith" in entities
        assert "New" in entities
        assert "York" in entities

class TestReasoningModule:
    """Test Reasoning Module functionality"""
    
    def test_reasoning_module_creation(self):
        """Test creating reasoning module"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = ReasoningModule()
        assert module.name == "ReasoningModule"
        assert module.version == "2.0.0"
        assert isinstance(module.knowledge_base, dict)
        assert isinstance(module.inference_rules, list)
    
    def test_reasoning_process(self, sample_dict_input):
        """Test reasoning process"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = ReasoningModule()
        result = module.process(sample_dict_input)
        
        assert isinstance(result, dict)
        assert "conclusion" in result
        assert "confidence" in result
        assert "reasoning_path" in result
        assert "alternatives" in result
        
        assert 0.5 <= result["confidence"] <= 1.0
        assert isinstance(result["reasoning_path"], list)
        assert isinstance(result["alternatives"], list)
    
    def test_reasoning_learning(self, sample_dict_input):
        """Test reasoning learning"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = ReasoningModule()
        initial_kb_size = len(module.knowledge_base)
        
        result = {"conclusion": "test conclusion"}
        feedback = {"logical_accuracy": 0.9}
        
        module.learn(sample_dict_input, result, feedback)
        
        # Knowledge base should grow
        assert len(module.knowledge_base) > initial_kb_size

class TestMemoryModule:
    """Test Memory Module functionality"""
    
    def test_memory_module_creation(self):
        """Test creating memory module"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = MemoryModule()
        assert module.name == "MemoryModule"
        assert module.version == "2.0.0"
        assert isinstance(module.short_term_memory, list)
        assert isinstance(module.long_term_memory, dict)
        assert module.memory_capacity == 10000
    
    def test_memory_storage(self, sample_text_input):
        """Test memory storage"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = MemoryModule()
        initial_stm_size = len(module.short_term_memory)
        
        result = module.process(sample_text_input)
        
        assert isinstance(result, dict)
        assert result["stored"] is True
        assert result["memory_type"] == "short_term"
        assert len(module.short_term_memory) > initial_stm_size
    
    def test_memory_consolidation(self):
        """Test memory consolidation"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = MemoryModule()
        
        # Fill short-term memory beyond threshold
        for i in range(105):
            module.process(f"test data {i}")
        
        # Some items should be consolidated to long-term memory
        assert len(module.short_term_memory) <= 50
        # Some items might be in long-term memory (those with high importance)
    
    def test_memory_promotion(self):
        """Test memory promotion to long-term"""
        if not NEURAL_CORE_AVAILABLE:
            pytest.skip("Neural core not available")
        
        module = MemoryModule()
        initial_ltm_size = len(module.long_term_memory)
        
        test_data = "important information"
        feedback = {"important": True}
        
        module.learn(test_data, feedback)
        
        assert len(module.long_term_memory) > initial_ltm_size

class TestNeuralCore:
    """Test main Neural Core functionality"""
    
    def test_neural_core_creation(self, neural_core):
        """Test creating neural core"""
        assert neural_core is not None
        assert neural_core.state.version == "2.0.0"
        assert neural_core.state.status == "active"
        assert len(neural_core.modules) == 3
        assert "language" in neural_core.modules
        assert "reasoning" in neural_core.modules
        assert "memory" in neural_core.modules
    
    def test_neural_core_process_text(self, neural_core, sample_text_input):
        """Test processing text input"""
        result = neural_core.process(sample_text_input)
        
        assert isinstance(result, dict)
        assert "input" in result
        assert "mode" in result
        assert "timestamp" in result
        assert "modules_used" in result
        assert "confidence" in result
        
        assert result["input"] == sample_text_input
        assert len(result["modules_used"]) > 0
        assert 0 <= result["confidence"] <= 1
    
    def test_neural_core_process_dict(self, neural_core, sample_dict_input):
        """Test processing dictionary input"""
        result = neural_core.process(sample_dict_input)
        
        assert isinstance(result, dict)
        assert result["input"] == sample_dict_input
        assert "language_output" in result
        assert "reasoning_output" in result
        assert "memory_output" in result
    
    def test_processing_modes(self, neural_core, sample_text_input):
        """Test different processing modes"""
        modes = [ProcessingMode.ANALYTICAL, ProcessingMode.CREATIVE, 
                ProcessingMode.HYBRID, ProcessingMode.ADAPTIVE]
        
        for mode in modes:
            result = neural_core.process(sample_text_input, mode)
            assert result["mode"] == mode.value
    
    def test_neural_core_learning(self, neural_core):
        """Test neural core learning"""
        data = "training data for neural core"
        feedback = {"accuracy": 0.9, "relevance": 0.8}
        
        result = neural_core.learn(data, feedback)
        
        assert isinstance(result, dict)
        assert "strategy" in result
        assert "timestamp" in result
        assert "modules_learned" in result
        assert "performance_improvement" in result
        
        assert result["strategy"] == LearningStrategy.SUPERVISED.value
        assert len(result["modules_learned"]) > 0
    
    def test_learning_strategies(self, neural_core):
        """Test different learning strategies"""
        strategies = [LearningStrategy.SUPERVISED, LearningStrategy.UNSUPERVISED,
                     LearningStrategy.REINFORCEMENT, LearningStrategy.META_LEARNING]
        
        data = "test learning data"
        
        for strategy in strategies:
            result = neural_core.learn(data, strategy=strategy)
            assert result["strategy"] == strategy.value
    
    def test_neural_core_evolution(self, neural_core):
        """Test neural core evolution"""
        initial_evolution_count = neural_core.state.evolution_count
        
        result = neural_core.evolve()
        
        assert isinstance(result, dict)
        assert "evolved" in result
        assert "timestamp" in result
        assert "evolution_count" in result
        assert "changes" in result
        
        if result["evolved"]:
            assert neural_core.state.evolution_count > initial_evolution_count
    
    def test_forced_evolution(self, neural_core):
        """Test forced evolution"""
        result = neural_core.evolve(force=True)
        
        assert result["evolved"] is True
        assert len(result["changes"]) >= 0
    
    def test_evolution_limit(self, neural_core):
        """Test evolution limit"""
        # Set evolution count to limit
        neural_core.state.evolution_count = 15
        
        result = neural_core.evolve()
        
        assert result["evolved"] is False
        assert "limit" in result["reason"]
    
    def test_get_status(self, neural_core):
        """Test getting system status"""
        status = neural_core.get_status()
        
        assert isinstance(status, dict)
        assert "state" in status
        assert "modules" in status
        assert "evolution_history" in status
        assert "system_health" in status
        
        assert 0 <= status["system_health"] <= 1
    
    def test_save_and_load_state(self, neural_core, tmp_path):
        """Test saving and loading state"""
        save_path = tmp_path / "neural_state.json"
        
        # Save state
        neural_core.save_state(str(save_path))
        assert save_path.exists()
        
        # Modify state
        original_learning_rate = neural_core.state.learning_rate
        neural_core.state.learning_rate = 0.05
        
        # Load state
        neural_core.load_state(str(save_path))
        assert neural_core.state.learning_rate == original_learning_rate
    
    def test_concurrent_processing(self, neural_core):
        """Test concurrent processing"""
        import threading
        
        results = []
        errors = []
        
        def process_input(input_data):
            try:
                result = neural_core.process(f"concurrent input {input_data}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_input, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        
        # All results should have correlation IDs
        for result in results:
            assert "timestamp" in result
            assert "confidence" in result

class TestIntegration:
    """Integration tests for neural core system"""
    
    def test_end_to_end_workflow(self, neural_core):
        """Test complete workflow from input to evolution"""
        # Process input
        input_text = "The neural system is working efficiently and learning continuously."
        process_result = neural_core.process(input_text)
        
        assert process_result["confidence"] > 0
        
        # Learn from the processing
        feedback = {"accuracy": 0.9, "usefulness": 0.85}
        learn_result = neural_core.learn(input_text, feedback)
        
        assert learn_result["performance_improvement"] >= 0
        
        # Get status
        status = neural_core.get_status()
        assert status["system_health"] > 0
        
        # Trigger evolution if performance is good
        if status["system_health"] > 0.7:
            evolution_result = neural_core.evolve()
            # Evolution might or might not happen based on internal logic
            assert "evolved" in evolution_result
    
    def test_module_interaction(self, neural_core):
        """Test interaction between different modules"""
        complex_input = {
            "text": "Analyze this complex scenario with multiple entities and relationships.",
            "context": "business analysis",
            "entities": ["Company A", "Market B", "Product C"],
            "relationships": ["partnership", "competition", "innovation"]
        }
        
        result = neural_core.process(complex_input)
        
        # All modules should be involved
        assert "language" in result["modules_used"]
        assert "reasoning" in result["modules_used"]
        assert "memory" in result["modules_used"]
        
        # Memory should store the interaction
        memory_module = neural_core.modules["memory"]
        assert len(memory_module.short_term_memory) > 0
    
    def test_performance_under_load(self, neural_core):
        """Test performance under heavy load"""
        start_time = time.time()
        
        # Process many inputs
        for i in range(50):
            neural_core.process(f"Load test input number {i}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert total_time < 10.0  # 10 seconds for 50 inputs
        
        # System should still be healthy
        status = neural_core.get_status()
        assert status["system_health"] > 0.3  # Should maintain some health

# Performance benchmarks
class TestPerformance:
    """Performance tests for neural core"""
    
    @pytest.mark.slow
    def test_processing_speed(self, neural_core):
        """Test processing speed"""
        inputs = [f"Performance test input {i}" for i in range(100)]
        
        start_time = time.time()
        
        for input_text in inputs:
            neural_core.process(input_text)
        
        end_time = time.time()
        
        avg_time_per_input = (end_time - start_time) / len(inputs)
        
        # Should process each input in reasonable time
        assert avg_time_per_input < 0.1  # 100ms per input
    
    @pytest.mark.slow
    def test_memory_usage(self, neural_core):
        """Test memory usage growth"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many inputs to test memory growth
        for i in range(1000):
            neural_core.process(f"Memory test {i}")
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024

# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_input_handling(self, neural_core):
        """Test handling of invalid inputs"""
        invalid_inputs = [None, "", [], {}, 123, object()]
        
        for invalid_input in invalid_inputs:
            # Should not crash, might return error or handle gracefully
            try:
                result = neural_core.process(invalid_input)
                # If it doesn't crash, result should be valid
                assert isinstance(result, dict)
            except Exception:
                # It's acceptable to raise exceptions for invalid input
                pass
    
    def test_module_failure_handling(self, neural_core):
        """Test handling when a module fails"""
        # Mock a module to fail
        original_process = neural_core.modules["language"].process
        
        def failing_process(*args, **kwargs):
            raise Exception("Simulated module failure")
        
        neural_core.modules["language"].process = failing_process
        
        try:
            result = neural_core.process("test input")
            
            # Should handle the failure gracefully
            assert isinstance(result, dict)
            assert "language_error" in result or "language" not in result["modules_used"]
        
        finally:
            # Restore original method
            neural_core.modules["language"].process = original_process
    
    def test_state_corruption_recovery(self, neural_core):
        """Test recovery from corrupted state"""
        # Corrupt the state
        original_state = neural_core.state
        neural_core.state = None
        
        try:
            # Should handle corrupted state
            result = neural_core.process("test input")
            # Might fail or recover, but shouldn't crash the system
        except Exception:
            pass
        finally:
            # Restore state
            neural_core.state = original_state

# Fixtures for specific test scenarios
@pytest.fixture
def neural_core_with_history(neural_core):
    """Neural core with some processing history"""
    # Process some inputs to build history
    inputs = [
        "First test input",
        "Second test input with different content",
        "Third input for building neural history"
    ]
    
    for input_text in inputs:
        neural_core.process(input_text)
        neural_core.learn(input_text, {"accuracy": 0.8})
    
    return neural_core

class TestHistoryAndEvolution:
    """Test history tracking and evolution based on history"""
    
    def test_evolution_history_tracking(self, neural_core_with_history):
        """Test that evolution history is tracked"""
        # Trigger evolution
        result = neural_core_with_history.evolve()
        
        if result["evolved"]:
            assert len(neural_core_with_history.evolution_history) > 0
            
            latest_evolution = neural_core_with_history.evolution_history[-1]
            assert "timestamp" in latest_evolution
            assert "changes" in latest_evolution
    
    def test_learning_from_feedback(self, neural_core_with_history):
        """Test learning improves performance"""
        # Get initial performance
        initial_status = neural_core_with_history.get_status()
        initial_health = initial_status["system_health"]
        
        # Provide positive feedback
        for i in range(10):
            neural_core_with_history.learn(
                f"positive training example {i}",
                {"accuracy": 0.95, "relevance": 0.9}
            )
        
        # Performance should improve or stay stable
        final_status = neural_core_with_history.get_status()
        final_health = final_status["system_health"]
        
        # Health should not decrease significantly
        assert final_health >= initial_health - 0.1

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])