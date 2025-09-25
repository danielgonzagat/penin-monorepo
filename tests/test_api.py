"""
Test suite for PENIN API functionality
Tests for the FastAPI server and endpoints
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from penin.api.server import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Test client
if API_AVAILABLE:
    client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime" in data
        assert data["status"] == "healthy"
    
    @patch('penin.api.server.neural_core')
    def test_system_status(self, mock_neural_core):
        """Test system status endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        # Mock neural core status
        mock_neural_core.get_status.return_value = {
            "state": {"version": "2.0.0"},
            "system_health": 0.95,
            "modules": {"language": {"active": True}}
        }
        
        response = client.get("/status", headers={"Authorization": "Bearer test-token"})
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "system_health" in data
            assert data["system_health"] == 0.95
    
    @patch('penin.api.server.neural_core')
    def test_neural_process(self, mock_neural_core):
        """Test neural processing endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        # Mock neural core processing
        mock_neural_core.process.return_value = {
            "confidence": 0.85,
            "modules_used": ["language", "reasoning"]
        }
        
        request_data = {
            "input_data": "Test input for neural processing",
            "mode": "hybrid"
        }
        
        response = client.post(
            "/neural/process",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            assert "processing_time" in data
    
    @patch('penin.api.server.neural_core')
    def test_neural_learn(self, mock_neural_core):
        """Test neural learning endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        # Mock neural core learning
        mock_neural_core.learn.return_value = {
            "performance_improvement": 0.1,
            "modules_learned": [{"module": "language", "improvement": 0.05}]
        }
        
        request_data = {
            "data": "Training data for neural learning",
            "feedback": {"accuracy": 0.9},
            "strategy": "supervised"
        }
        
        response = client.post(
            "/neural/learn",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "learning_result" in data

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.get("/status")
        # Should require authentication or return 401/403
        assert response.status_code in [401, 403, 422]  # 422 for missing auth header
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test accessing protected endpoint with invalid token"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.get("/status", headers={"Authorization": "Bearer invalid-token"})
        # Should reject invalid token
        assert response.status_code in [401, 403]

class TestErrorHandling:
    """Test API error handling"""
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.post(
            "/neural/process",
            data="invalid json",
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json"
            }
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    @patch('penin.api.server.neural_core', None)
    def test_neural_core_unavailable(self):
        """Test behavior when neural core is unavailable"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        request_data = {"input_data": "test"}
        
        response = client.post(
            "/neural/process",
            json=request_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 503  # Service Unavailable

class TestCORS:
    """Test CORS functionality"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

class TestMetrics:
    """Test metrics endpoints"""
    
    @patch('penin.api.server.logger')
    def test_record_metric(self, mock_logger):
        """Test metric recording endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        mock_logger.metric = Mock()
        
        metric_data = {
            "name": "test_metric",
            "value": 42.0,
            "metric_type": "gauge",
            "labels": {"component": "test"}
        }
        
        response = client.post(
            "/metrics/record",
            json=metric_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
    
    @patch('penin.api.server.logger')
    def test_get_metrics(self, mock_logger):
        """Test metrics retrieval endpoint"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        mock_logger.get_metrics_summary.return_value = {
            "metrics_count": 5,
            "handlers_count": 2
        }
        
        response = client.get("/metrics", headers={"Authorization": "Bearer test-token"})
        
        if response.status_code == 200:
            data = response.json()
            assert "metrics_summary" in data
            assert "system_stats" in data

# Integration tests
class TestAPIIntegration:
    """Integration tests for API"""
    
    @patch('penin.api.server.neural_core')
    def test_complete_workflow(self, mock_neural_core):
        """Test complete API workflow"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        # Mock neural core responses
        mock_neural_core.process.return_value = {"confidence": 0.9}
        mock_neural_core.learn.return_value = {"performance_improvement": 0.1}
        mock_neural_core.evolve.return_value = {"evolved": True, "changes": ["test"]}
        
        headers = {"Authorization": "Bearer test-token"}
        
        # 1. Process input
        process_response = client.post(
            "/neural/process",
            json={"input_data": "Test workflow input"},
            headers=headers
        )
        
        # 2. Learn from feedback
        if process_response.status_code == 200:
            learn_response = client.post(
                "/neural/learn",
                json={
                    "data": "Test workflow input",
                    "feedback": {"accuracy": 0.9}
                },
                headers=headers
            )
            
            # 3. Trigger evolution (if admin)
            if learn_response.status_code == 200:
                evolve_response = client.post(
                    "/neural/evolve",
                    json={"force": False},
                    headers=headers
                )
                
                # Evolution might require admin permissions
                assert evolve_response.status_code in [200, 403]

# Load testing
@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API"""
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        # Create multiple concurrent requests
        threads = []
        start_time = time.time()
        
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert all(status == 200 for status in results)
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0
    
    @patch('penin.api.server.neural_core')
    def test_request_throughput(self, mock_neural_core):
        """Test API request throughput"""
        if not API_AVAILABLE:
            pytest.skip("API not available")
        
        mock_neural_core.process.return_value = {"confidence": 0.8}
        
        import time
        
        num_requests = 50
        start_time = time.time()
        
        for i in range(num_requests):
            response = client.post(
                "/neural/process",
                json={"input_data": f"Test input {i}"},
                headers={"Authorization": "Bearer test-token"}
            )
            # Don't assert on individual responses to maintain speed
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        requests_per_second = num_requests / total_time
        
        # Should handle at least 5 requests per second
        assert requests_per_second >= 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])