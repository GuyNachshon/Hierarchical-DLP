"""
Entry point for the modular agentic data generator.
This file provides backward compatibility by importing from the new modular structure.
"""

import sys
from pathlib import Path

# Import from the modular structure
try:
    from .agentic_data_generator.main import run, main
    from .agentic_data_generator.config import AgenticConfig, create_demo_config, create_production_config
    from .agentic_data_generator.coordinator import AgenticDataGenerator
    
    # Re-export for backward compatibility
    __all__ = ['run', 'main', 'AgenticConfig', 'create_demo_config', 'create_production_config', 'AgenticDataGenerator']
    
except ImportError as e:
    print(f"❌ Failed to import modular components: {e}")
    print("💡 Make sure all dependencies are properly set up")
    sys.exit(1)


if __name__ == "__main__":
    run()