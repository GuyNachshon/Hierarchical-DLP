"""
DLP Model - HRM Extension for Data Loss Prevention

Multi-task model extending HRM with document classification and span tagging heads.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Import from HRM directory for now (legacy compatibility)
import sys
from pathlib import Path
hrm_path = Path(__file__).parent.parent.parent / "HRM"
sys.path.insert(0, str(hrm_path))

from hrm_dlp.model import HRMDLPModel
from models.hrm.hrm_act_v1 import HRMConfig

@dataclass
class DLPModelConfig:
    """Configuration for DLP model"""
    vocab_size: int = 16000
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    num_doc_labels: int = 4
    num_bio_tags: int = 21
    enable_fusion_gates: bool = True
    max_position_embeddings: int = 1024
    intermediate_size: int = 1536
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1


class DLPModel(nn.Module):
    """DLP Model extending HRM for multi-task learning"""
    
    def __init__(self, config: DLPModelConfig):
        super().__init__()
        self.config = config
        
        # Create HRM config from DLP config
        hrm_config = HRMConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_position_embeddings=config.max_position_embeddings,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_dropout_prob=config.attention_dropout_prob,
        )
        
        # Use existing HRM-DLP model from legacy location
        self.hrm_dlp = HRMDLPModel(
            hrm_config,
            num_doc_labels=config.num_doc_labels,
            num_bio_tags=config.num_bio_tags
        )
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[Any, Any]:
        """Forward pass through the model"""
        return self.hrm_dlp(input_ids, **kwargs)