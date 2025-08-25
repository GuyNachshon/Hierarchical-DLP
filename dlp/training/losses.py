"""
DLP Loss Functions

Multi-task loss functions for DLP training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path
import sys

# Legacy import compatibility
hrm_path = Path(__file__).parent.parent.parent / "HRM"
sys.path.insert(0, str(hrm_path))

from hrm_dlp.losses import MultiTaskDLPLoss


def create_dlp_loss(
    vocab_size: int,
    config_dict: Dict[str, Any],
    num_doc_labels: int = 4,
    num_bio_tags: int = 21,
    hidden_size: int = 384
) -> nn.Module:
    """Create DLP loss function"""
    
    return MultiTaskDLPLoss(
        vocab_size=vocab_size,
        num_doc_labels=num_doc_labels,
        num_bio_tags=num_bio_tags,
        hidden_size=hidden_size,
        **config_dict
    )