#!/usr/bin/env python3
"""
Neural Core Demonstration Script

This script demonstrates the optimized Neural Core functionality,
showcasing performance improvements, caching, error handling, and logging.
"""

import time
from neural_core import NeuralCore, SystemStatus


def main():
    """Main demonstration function."""
    print("🧠 PENIN Code Optimizer Agent - Neural Core Demo")
    print("=" * 50)
    
    # Initialize Neural Core
    print("\n1. Initializing Neural Core...")
    core = NeuralCore(version="1.0.0", cache_size=5)
    
    # Display system info
    print("\n2. System Information:")
    info = core.get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Demonstrate processing with caching
    print("\n3. Testing Processing & Caching:")
    test_data = ["hello", "world", "neural", "core", "hello"]  # "hello" repeated
    
    for i, data in enumerate(test_data, 1):
        result = core.process(data)
        cached_status = "CACHED" if result.metadata.get("cached") else "NEW"
        print(f"   {i}. '{data}' -> {result.result} "
              f"[{cached_status}, {result.processing_time:.4f}s]")
    
    # Demonstrate learning
    print("\n4. Testing Learning:")
    learning_data = [
        ["pattern1", "pattern2", "pattern3"],
        {"key1": "value1", "key2": "value2"},
        "single_pattern"
    ]
    
    for i, data in enumerate(learning_data, 1):
        success = core.learn(data)
        print(f"   {i}. Learning from {type(data).__name__}: {'SUCCESS' if success else 'FAILED'}")
    
    # Display updated system info
    print("\n5. Updated System Information:")
    info = core.get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Demonstrate evolution
    print("\n6. Testing Evolution:")
    for i in range(3):
        evolution_result = core.evolve()
        print(f"   Evolution #{evolution_result['evolution_count']}: "
              f"{len(evolution_result['optimizations_applied'])} optimizations applied")
        if evolution_result['optimizations_applied']:
            print(f"      Applied: {', '.join(evolution_result['optimizations_applied'])}")
    
    # Performance test
    print("\n7. Performance Test:")
    start_time = time.perf_counter()
    
    # Process 100 items
    for i in range(100):
        core.process(f"performance_test_{i % 10}")  # Some repetition for cache testing
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print(f"   Processed 100 items in {total_time:.4f}s")
    print(f"   Average time per item: {total_time/100:.6f}s")
    
    # Final system state
    print("\n8. Final System State:")
    final_info = core.get_system_info()
    for key, value in final_info.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"Neural Core: {core}")


if __name__ == "__main__":
    main()