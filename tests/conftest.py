"""
Pytest configuration and shared fixtures
Global test configuration for the PENIN system
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "neural: marks tests related to neural core"
    )
    config.addinivalue_line(
        "markers", "api: marks tests related to API"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests related to machine learning"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test file names
    for item in items:
        if "test_neural" in item.nodeid:
            item.add_marker(pytest.mark.neural)
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_ml" in item.nodeid or "test_model" in item.nodeid:
            item.add_marker(pytest.mark.ml)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

# Global fixtures
@pytest.fixture(scope="session")
def test_workspace():
    """Create a temporary workspace for tests"""
    temp_dir = tempfile.mkdtemp(prefix="penin_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    logger.metric = Mock()
    logger.timer = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
    logger.correlation = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
    logger.get_metrics_summary = Mock(return_value={})
    return logger

@pytest.fixture
def mock_config():
    """Mock configuration manager for testing"""
    config = Mock()
    config.get = Mock(return_value=None)
    config.set = Mock()
    config.get_neural_config = Mock(return_value={})
    config.get_api_config = Mock(return_value={})
    config.get_ml_config = Mock(return_value={})
    return config

@pytest.fixture
def sample_texts():
    """Sample text data for testing"""
    return [
        "This is a positive example of text processing.",
        "The neural network is working efficiently and accurately.",
        "Machine learning models require proper training data.",
        "The system performance has improved significantly.",
        "Testing is crucial for software quality assurance."
    ]

@pytest.fixture
def sample_training_data():
    """Sample training data for ML models"""
    return {
        "texts": [
            "Positive sentiment example",
            "Negative sentiment example", 
            "Neutral statement about facts",
            "Very positive and enthusiastic text",
            "Somewhat negative but not extreme"
        ],
        "labels": [1, 0, 2, 1, 0],  # 0=negative, 1=positive, 2=neutral
        "categories": ["positive", "negative", "neutral", "positive", "negative"]
    }

@pytest.fixture
def mock_neural_core():
    """Mock neural core for testing"""
    neural_core = Mock()
    
    # Mock state
    neural_core.state = Mock()
    neural_core.state.version = "2.0.0"
    neural_core.state.status = "active"
    neural_core.state.evolution_count = 0
    neural_core.state.learning_rate = 0.01
    neural_core.state.confidence_threshold = 0.7
    
    # Mock modules
    neural_core.modules = {
        "language": Mock(),
        "reasoning": Mock(),
        "memory": Mock()
    }
    
    # Mock methods
    neural_core.process = Mock(return_value={
        "confidence": 0.85,
        "modules_used": ["language", "reasoning"],
        "timestamp": "2025-01-01T00:00:00"
    })
    
    neural_core.learn = Mock(return_value={
        "performance_improvement": 0.1,
        "modules_learned": [{"module": "language", "improvement": 0.05}]
    })
    
    neural_core.evolve = Mock(return_value={
        "evolved": True,
        "changes": ["test change"],
        "timestamp": "2025-01-01T00:00:00"
    })
    
    neural_core.get_status = Mock(return_value={
        "system_health": 0.95,
        "state": {"version": "2.0.0"},
        "modules": {"language": {"active": True}}
    })
    
    return neural_core

@pytest.fixture
def mock_evolution_engine():
    """Mock evolution engine for testing"""
    engine = Mock()
    
    engine.scan_system = Mock(return_value={
        "files_analyzed": 10,
        "issues_found": 5,
        "optimization_opportunities": 3
    })
    
    engine.generate_evolution_plan = Mock(return_value=[])
    engine.execute_evolution_plan = Mock(return_value=Mock(success=True))
    engine.auto_evolve = Mock(return_value=[])
    engine.get_evolution_status = Mock(return_value={
        "stats": {"total_evolutions": 0},
        "auto_evolution_enabled": True
    })
    
    return engine

@pytest.fixture
def test_config_file(test_workspace):
    """Create a test configuration file"""
    config_content = """
system:
  name: "PENIN Test System"
  version: "2.0.0"
  environment: "test"

neural_core:
  version: "2.0.0"
  processing_mode: "hybrid"
  learning_rate: 0.01

api:
  host: "127.0.0.1"
  port: 8001
  workers: 1

logging:
  level: "DEBUG"
  file_path: null
"""
    
    config_file = test_workspace / "test_config.yaml"
    config_file.write_text(config_content)
    
    return config_file

@pytest.fixture
def test_model_data():
    """Test data for ML models"""
    return {
        "input_texts": [
            "The weather is beautiful today",
            "I hate this terrible situation",
            "The report shows neutral results",
            "Absolutely amazing performance!",
            "This is disappointing news"
        ],
        "expected_sentiments": ["positive", "negative", "neutral", "positive", "negative"],
        "expected_classifications": [1, 0, 2, 1, 0]
    }

# Test utilities
class TestUtils:
    """Utility functions for tests"""
    
    @staticmethod
    def create_mock_response(status_code=200, json_data=None):
        """Create mock HTTP response"""
        response = Mock()
        response.status_code = status_code
        response.json = Mock(return_value=json_data or {})
        response.text = str(json_data) if json_data else ""
        return response
    
    @staticmethod
    def assert_valid_timestamp(timestamp_str):
        """Assert that a string is a valid ISO timestamp"""
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False
    
    @staticmethod
    def assert_valid_uuid(uuid_str):
        """Assert that a string is a valid UUID"""
        import uuid
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False

@pytest.fixture
def test_utils():
    """Test utilities fixture"""
    return TestUtils

# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_deps(monkeypatch):
    """Mock external dependencies that might not be available"""
    
    # Mock transformers if not available
    try:
        import transformers
    except ImportError:
        mock_transformers = Mock()
        mock_transformers.AutoTokenizer = Mock()
        mock_transformers.AutoModel = Mock()
        mock_transformers.BertModel = Mock()
        mock_transformers.BertTokenizer = Mock()
        monkeypatch.setattr("sys.modules.transformers", mock_transformers)
    
    # Mock torch if not available
    try:
        import torch
    except ImportError:
        mock_torch = Mock()
        mock_torch.nn = Mock()
        mock_torch.optim = Mock()
        mock_torch.tensor = Mock()
        monkeypatch.setattr("sys.modules.torch", mock_torch)
    
    # Mock numpy if not available (unlikely but just in case)
    try:
        import numpy
    except ImportError:
        mock_numpy = Mock()
        mock_numpy.random = Mock()
        mock_numpy.mean = Mock(return_value=0.5)
        monkeypatch.setattr("sys.modules.numpy", mock_numpy)

# Database fixtures for integration tests
@pytest.fixture
def test_database():
    """In-memory database for testing"""
    # This would set up a test database
    # For now, return a mock
    db = Mock()
    db.connect = Mock()
    db.execute = Mock()
    db.close = Mock()
    return db

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("PENIN_ENV", "test")
    monkeypatch.setenv("PENIN_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PENIN_TESTING", "true")

# Async fixtures for async tests
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor for performance testing"""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            return {
                "duration": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "peak_memory": self.process.memory_info().rss
            }
    
    return PerformanceMonitor()

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test"""
    yield
    # Cleanup code here if needed
    # For example, clear caches, reset global state, etc.
    pass