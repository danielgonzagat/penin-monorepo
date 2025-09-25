"""
Neural Core - Núcleo do sistema ET Ultimate

Este módulo implementa o núcleo neural do sistema ET Ultimate,
fornecendo funcionalidades de processamento, aprendizagem e evolução.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SystemStatus(Enum):
    """Enumeration for system status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LEARNING = "learning"
    EVOLVING = "evolving"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Data class for processing results."""
    result: Any
    processing_time: float
    status: SystemStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuralInterface(ABC):
    """Abstract interface for neural processing components."""
    
    @abstractmethod
    def process(self, input_data: Any) -> ProcessingResult:
        """Process input data and return result."""
        pass
    
    @abstractmethod
    def learn(self, data: Union[List[Any], Dict[str, Any]]) -> bool:
        """Learn from provided data."""
        pass


class NeuralCore(NeuralInterface):
    """
    Neural Core - Main processing unit of the ET Ultimate system.
    
    This class provides the core functionality for data processing,
    machine learning, and system evolution. It maintains state,
    handles errors gracefully, and provides comprehensive logging.
    
    Attributes:
        version (str): Current version of the neural core
        status (SystemStatus): Current operational status
        learning_data (List[Any]): Accumulated learning data
        processing_cache (Dict[str, Any]): Cache for processed results
    """
    
    def __init__(self, version: str = "1.0.0", cache_size: int = 1000) -> None:
        """
        Initialize the Neural Core.
        
        Args:
            version: Version string for the neural core
            cache_size: Maximum size of the processing cache
            
        Raises:
            ValueError: If cache_size is negative
        """
        if cache_size < 0:
            raise ValueError("Cache size must be non-negative")
            
        self.version: str = version
        self.status: SystemStatus = SystemStatus.ACTIVE
        self.learning_data: List[Any] = []
        self.processing_cache: Dict[str, Any] = {}
        self._cache_size: int = cache_size
        self._evolution_count: int = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Neural Core v{self.version} initialized successfully")
    
    def process(self, input_data: Any) -> ProcessingResult:
        """
        Process input data and return comprehensive results.
        
        This method handles data processing with caching, error handling,
        and performance monitoring. Results are cached for efficiency.
        
        Args:
            input_data: Data to be processed (any type)
            
        Returns:
            ProcessingResult: Comprehensive processing results including
                             timing, status, and metadata
                             
        Raises:
            RuntimeError: If system is in error state
            TypeError: If input_data is None
        """
        if self.status == SystemStatus.ERROR:
            raise RuntimeError("Cannot process data: system is in error state")
        
        if input_data is None:
            raise TypeError("Input data cannot be None")
        
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = str(hash(str(input_data)))
            if cache_key in self.processing_cache:
                self.logger.debug(f"Cache hit for input: {input_data}")
                cached_result = self.processing_cache[cache_key]
                processing_time = time.perf_counter() - start_time
                
                return ProcessingResult(
                    result=cached_result,
                    processing_time=processing_time,
                    status=self.status,
                    metadata={"cached": True, "cache_key": cache_key}
                )
            
            # Process the data
            processed_result = f"Processado: {input_data}"
            processing_time = time.perf_counter() - start_time
            
            # Cache the result if cache isn't full
            if len(self.processing_cache) < self._cache_size:
                self.processing_cache[cache_key] = processed_result
            elif self.processing_cache:  # Remove oldest entry if cache is full
                oldest_key = next(iter(self.processing_cache))
                del self.processing_cache[oldest_key]
                self.processing_cache[cache_key] = processed_result
            
            self.logger.info(
                f"Processed data in {processing_time:.4f}s: {str(input_data)[:50]}..."
            )
            
            return ProcessingResult(
                result=processed_result,
                processing_time=processing_time,
                status=self.status,
                metadata={
                    "cached": False,
                    "cache_key": cache_key,
                    "cache_size": len(self.processing_cache)
                }
            )
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Processing failed: {str(e)}")
            processing_time = time.perf_counter() - start_time
            
            return ProcessingResult(
                result=None,
                processing_time=processing_time,
                status=SystemStatus.ERROR,
                metadata={"error": str(e)}
            )
    
    def learn(self, data: Union[List[Any], Dict[str, Any]]) -> bool:
        """
        Learn from provided data and update internal knowledge.
        
        This method accumulates learning data and updates the system's
        knowledge base. It handles various data formats and provides
        feedback on the learning process.
        
        Args:
            data: Learning data (list, dictionary, or other formats)
            
        Returns:
            bool: True if learning was successful, False otherwise
            
        Raises:
            ValueError: If data is empty or invalid format
        """
        if not data:
            raise ValueError("Learning data cannot be empty")
        
        previous_status = self.status
        self.status = SystemStatus.LEARNING
        
        try:
            # Normalize data format
            if isinstance(data, dict):
                learning_items = list(data.values())
            elif isinstance(data, list):
                learning_items = data
            else:
                learning_items = [data]
            
            # Add to learning data
            self.learning_data.extend(learning_items)
            
            # Limit learning data size to prevent memory issues
            max_learning_data = 10000
            if len(self.learning_data) > max_learning_data:
                self.learning_data = self.learning_data[-max_learning_data:]
                self.logger.warning(
                    f"Learning data truncated to {max_learning_data} items"
                )
            
            self.logger.info(
                f"Learned from {len(learning_items)} items. "
                f"Total learning data: {len(self.learning_data)}"
            )
            
            self.status = previous_status
            return True
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Learning failed: {str(e)}")
            return False
    
    def evolve(self) -> Dict[str, Any]:
        """
        Perform system evolution and optimization.
        
        This method implements self-improvement mechanisms,
        optimizing performance and updating system parameters
        based on accumulated learning data.
        
        Returns:
            Dict[str, Any]: Evolution results and statistics
            
        Raises:
            RuntimeError: If system is in error state
        """
        if self.status == SystemStatus.ERROR:
            raise RuntimeError("Cannot evolve: system is in error state")
        
        previous_status = self.status
        self.status = SystemStatus.EVOLVING
        
        try:
            start_time = time.perf_counter()
            
            # Perform evolution steps
            evolution_results = {
                "evolution_count": self._evolution_count + 1,
                "learning_data_size": len(self.learning_data),
                "cache_efficiency": self._calculate_cache_efficiency(),
                "optimizations_applied": []
            }
            
            # Cache optimization
            if len(self.processing_cache) > self._cache_size * 0.8:
                # Clear old cache entries
                cache_to_remove = len(self.processing_cache) // 2
                keys_to_remove = list(self.processing_cache.keys())[:cache_to_remove]
                for key in keys_to_remove:
                    del self.processing_cache[key]
                evolution_results["optimizations_applied"].append("cache_cleanup")
            
            # Version increment for significant evolutions
            if self._evolution_count > 0 and self._evolution_count % 10 == 0:
                version_parts = self.version.split('.')
                if len(version_parts) == 3:
                    version_parts[2] = str(int(version_parts[2]) + 1)
                    self.version = '.'.join(version_parts)
                    evolution_results["optimizations_applied"].append("version_increment")
            
            self._evolution_count += 1
            evolution_time = time.perf_counter() - start_time
            evolution_results["evolution_time"] = evolution_time
            
            self.logger.info(
                f"Evolution #{self._evolution_count} completed in {evolution_time:.4f}s"
            )
            
            self.status = previous_status
            return evolution_results
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Evolution failed: {str(e)}")
            return {"error": str(e), "evolution_count": self._evolution_count}
    
    def _calculate_cache_efficiency(self) -> float:
        """
        Calculate cache efficiency metrics.
        
        Returns:
            float: Cache efficiency ratio (0.0 to 1.0)
        """
        if not self.processing_cache:
            return 0.0
        return min(len(self.processing_cache) / self._cache_size, 1.0)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information and statistics.
        
        Returns:
            Dict[str, Any]: System information including version,
                           status, performance metrics, and statistics
        """
        return {
            "version": self.version,
            "status": self.status.value,
            "evolution_count": self._evolution_count,
            "learning_data_size": len(self.learning_data),
            "cache_size": len(self.processing_cache),
            "cache_capacity": self._cache_size,
            "cache_efficiency": self._calculate_cache_efficiency()
        }
    
    def reset(self) -> None:
        """
        Reset the neural core to initial state.
        
        Clears all caches, learning data, and resets counters.
        Useful for testing or system maintenance.
        """
        self.processing_cache.clear()
        self.learning_data.clear()
        self._evolution_count = 0
        self.status = SystemStatus.ACTIVE
        self.logger.info("Neural Core reset to initial state")
    
    def __str__(self) -> str:
        """String representation of the Neural Core."""
        return f"NeuralCore(v{self.version}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the Neural Core."""
        return (
            f"NeuralCore(version='{self.version}', "
            f"status={self.status}, "
            f"cache_size={len(self.processing_cache)}, "
            f"learning_data_size={len(self.learning_data)})"
        )
