"""
DLP Models

DLP extensions to HRM with multi-task heads for document classification and span tagging.
"""

from .dlp_model import DLPModel, DLPModelConfig

def create_dlp_model(config):
    """Factory function to create DLP model"""
    return DLPModel(config)

__all__ = ["DLPModel", "DLPModelConfig", "create_dlp_model"]