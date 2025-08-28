"""
Enhanced Agentic Data Generator - A robust system for generating DLP training data.

A clean, modular system with enhanced batch management, run session isolation,
and robust recovery capabilities for generating DLP training data using a 3-tier agentic architecture.
"""

from .config import AgenticConfig, create_demo_config, create_production_config
from .coordinator import AgenticDataGenerator  # Legacy coordinator
from .enhanced_coordinator import EnhancedAgenticDataGenerator  # Enhanced coordinator
from .main import run, main

# Batch management system
from .batch.run_session_manager import RunSessionManager
from .batch.enhanced_batch_tracker import EnhancedBatchTracker
from .batch.connection_monitor import ConnectionMonitor
from .batch.split_batch_coordinator import SplitBatchCoordinator
from .batch.batch_recovery_manager import BatchRecoveryManager

__version__ = "2.1.0"
__all__ = [
    # Core system
    "AgenticConfig", 
    "create_demo_config", 
    "create_production_config",
    "AgenticDataGenerator",           # Legacy
    "EnhancedAgenticDataGenerator",   # Enhanced
    "run", 
    "main",
    
    # Batch management
    "RunSessionManager",
    "EnhancedBatchTracker", 
    "ConnectionMonitor",
    "SplitBatchCoordinator",
    "BatchRecoveryManager"
]