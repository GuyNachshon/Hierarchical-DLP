#!/usr/bin/env python3
"""
Debug where the startup is blocking
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

async def debug_startup():
    """Debug startup sequence step by step"""
    print("🔍 DEBUG: Starting debug sequence...")
    
    # Test 1: Basic imports
    print("🔍 DEBUG: Testing imports...")
    try:
        from agentic_data_generator import AgenticDataGenerator, AgenticConfig
        print("✅ DEBUG: Import successful")
    except Exception as e:
        print(f"❌ DEBUG: Import failed: {e}")
        return
    
    # Test 2: Config creation
    print("🔍 DEBUG: Testing config creation...")
    try:
        config = AgenticConfig(
            output_dir="/tmp/debug_test",
            train_size=10,
            val_size=5,
            test_size=5,
            max_concurrent_agents=2
        )
        print("✅ DEBUG: Config created successfully")
    except Exception as e:
        print(f"❌ DEBUG: Config creation failed: {e}")
        return
    
    # Test 3: Generator initialization (this might be where it blocks)
    print("🔍 DEBUG: Testing generator initialization...")
    try:
        generator = AgenticDataGenerator(config)
        print("✅ DEBUG: Generator initialized successfully")
    except Exception as e:
        print(f"❌ DEBUG: Generator initialization failed: {e}")
        return
    
    # Test 4: Task manager
    print("🔍 DEBUG: Testing task manager...")
    try:
        from task_dashboard import get_task_manager
        task_manager = get_task_manager()
        task_manager.create_task("debug_test", "Testing task creation")
        print("✅ DEBUG: Task manager working")
    except Exception as e:
        print(f"❌ DEBUG: Task manager failed: {e}")
        return
    
    # Test 5: Dashboard thread
    print("🔍 DEBUG: Testing dashboard thread startup...")
    try:
        from task_dashboard import start_dashboard_thread
        dashboard_thread = start_dashboard_thread()
        print("✅ DEBUG: Dashboard thread started")
        await asyncio.sleep(2)  # Give it time to start
        task_manager.shutdown()
        print("✅ DEBUG: Dashboard test completed")
    except Exception as e:
        print(f"❌ DEBUG: Dashboard failed: {e}")
        return
    
    print("🎉 DEBUG: All startup components working!")

if __name__ == "__main__":
    print("🔍 DEBUG: Script starting...")
    asyncio.run(debug_startup())
    print("🔍 DEBUG: Script completed")