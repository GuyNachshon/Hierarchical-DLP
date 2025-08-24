"""
Data Generation and Processing Module

Simplified data generation system for both puzzle datasets (ARC, Sudoku, Maze)
and synthetic DLP data, consolidated from the complex multi-agent system.
"""

from .generators import *
from .synthetic import *

__all__ = ["build_dataset", "generate_dlp_dataset", "SyntheticDataGenerator", "SimpleAgentSystem"]