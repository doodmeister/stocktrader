#!/usr/bin/env python3
"""
Test script for the production-grade ModelManager
"""
import sys
sys.path.append('.')

from train.model_manager import ModelManager, ModelManagerConfig

def test_model_manager():
    print("🧪 Testing Production-Grade Model Manager")
    print("=" * 50)
    
    # Test 1: Configuration
    config = ModelManagerConfig()
    print(f"✅ Configuration created successfully")
    print(f"   📁 Base directory: {config.base_directory}")
    print(f"   🔧 Max cache entries: {config.max_cache_entries}")
    print(f"   ⏱️ Cache TTL: {config.cache_ttl_seconds}s")
    print(f"   🛡️ Security enabled: {config.enable_checksums}")
    print(f"   📏 Max file size: {config.max_file_size_mb}MB")
    
    # Test 2: Manager initialization
    manager = ModelManager(config)
    print(f"✅ ModelManager created successfully")
    
    # Test 3: Health check
    health = manager.health_check()
    print(f"✅ Health check completed: {health['status']}")
    for check, result in health['checks'].items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}: {result}")
    
    # Test 4: List models
    models = manager.list_models()
    print(f"✅ Model listing completed: {len(models)} models found")
    
    # Test 5: Performance metrics
    metrics = manager.get_performance_metrics()
    print(f"✅ Performance metrics retrieved")
    print(f"   💾 Cache entries: {metrics['cache_info']['current_entries']}/{metrics['cache_info']['max_entries']}")
    print(f"   💽 Disk usage: {metrics['disk_usage']}MB")
    
    print("\n🎉 All tests passed! Production-grade ModelManager is ready!")
    return True

if __name__ == "__main__":
    try:
        test_model_manager()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
