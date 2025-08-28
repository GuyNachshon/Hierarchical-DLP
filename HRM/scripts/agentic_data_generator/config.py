"""
Clean, simplified configuration for agentic data generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgenticConfig:
    """Simplified configuration for agentic data generation."""
    
    # Output settings
    output_dir: str = "data/dlp_agentic"
    train_size: int = 2000
    val_size: int = 400
    test_size: int = 400
    
    # Model settings
    temperature: float = 0.8
    max_retries: int = 3
    max_concurrent_agents: int = 10
    
    # Batch processing - FIXED to use one batch per split
    batch_size: int = 20  # Legacy - no longer used for splitting
    batch_threshold: int = 50
    enable_batch_api: bool = True
    max_batch_size: int = 140000  # Maximum entries per batch (140K limit)
    
    # Batch behavior controls
    submit_splits_concurrently: bool = True  # Submit train/val/test in parallel
    auto_retrieve_batches: bool = True       # Wait and fetch batch results automatically
    
    # Quality control
    min_quality_score: float = 0.7
    thread_probability: float = 0.3
    max_thread_length: int = 5
    
    # Dataset balance
    agent_distribution: Optional[Dict[str, float]] = None
    risk_distribution: Optional[Dict[str, float]] = None
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_ratio: float = 0.3
    
    # State management
    enable_resume: bool = True
    state_save_interval: int = 100
    checkpoint_dir: Optional[str] = None
    
    # System
    seed: int = 42
    
    def __post_init__(self):
        """Set default distributions and paths."""
        if self.agent_distribution is None:
            self.agent_distribution = {
                "clean_business": 0.5,
                "casual": 0.25,
                "legal": 0.08,
                "finance": 0.08,
                "hr": 0.04,
                "security": 0.03,
                "obfuscation": 0.02
            }
        
        if self.risk_distribution is None:
            self.risk_distribution = {
                "no_risk": 0.40,
                "low_risk": 0.45,
                "medium_risk": 0.10,
                "high_risk": 0.04,
                "obfuscated": 0.01
            }
        
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.output_dir}/.checkpoints"


def create_demo_config() -> AgenticConfig:
    """Create a configuration for demo/testing purposes."""
    return AgenticConfig(
        output_dir="data/dlp_demo",
        train_size=100,
        val_size=20,
        test_size=20,
        max_concurrent_agents=5,
        batch_threshold=20,
        enable_batch_api=False  # Use concurrent for demo
    )


def create_production_config() -> AgenticConfig:
    """Create a configuration for production data generation."""
    return AgenticConfig(
        output_dir="data/dlp_production",
        train_size=60000,
        val_size=5000,
        test_size=5000,
        max_concurrent_agents=20,
        batch_threshold=100,
        enable_batch_api=True
    )


def create_hrm_dlp_config() -> AgenticConfig:
    """Create a configuration optimized for HRM-DLP training data generation."""
    return AgenticConfig(
        # HRM-DLP specific output
        output_dir="data/hrm_dlp_enhanced",
        train_size=2000,
        val_size=400, 
        test_size=400,
        
        # Enhanced quality for multi-task learning
        temperature=0.7,  # Slightly lower for more consistent quality
        max_retries=5,    # Higher retries for better reliability
        min_quality_score=0.8,  # Higher quality threshold
        
        # Batch processing optimized for GPT-5
        enable_batch_api=True,
        auto_retrieve_batches=True,
        submit_splits_concurrently=True,
        max_concurrent_agents=15,
        
        # Enhanced conversational patterns for HRM reasoning
        thread_probability=0.4,  # More multi-turn conversations
        max_thread_length=7,     # Longer reasoning chains
        
        # Balanced distribution for DLP scenarios
        agent_distribution={
            "legal": 0.20,           # High-value reasoning scenarios
            "finance": 0.20,         # Payment/banking scenarios
            "hr": 0.15,              # Employee data scenarios
            "security": 0.15,        # Incident/credential scenarios  
            "clean_business": 0.20,  # Normal business communications
            "casual": 0.08,          # Personal info leakage
            "obfuscation": 0.02      # Hidden content scenarios
        },
        
        # Risk distribution for multi-task learning
        risk_distribution={
            "no_risk": 0.30,         # Normal business (30%)
            "low_risk": 0.35,        # Minor concerns (35%)
            "medium_risk": 0.25,     # Clear issues (25%)
            "high_risk": 0.08,       # Serious violations (8%)
            "obfuscated": 0.02       # Hidden threats (2%)
        },
        
        # Enhanced augmentation for conversation diversity
        enable_augmentation=True,
        augmentation_ratio=2.0,  # 2x expansion with comprehensive variations
        
        # Robust state management
        enable_resume=True,
        state_save_interval=50,  # More frequent saves
        
        seed=42
    )
