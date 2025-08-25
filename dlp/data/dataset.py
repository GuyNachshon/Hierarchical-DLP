"""
DLP Dataset Loading

Dataset loading utilities for DLP training data.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

# Legacy import compatibility
import sys
hrm_path = Path(__file__).parent.parent.parent / "HRM"
sys.path.insert(0, str(hrm_path))

from hrm_dlp.dataset import DLPDataset as LegacyDLPDataset

@dataclass
class DLPDatasetConfig:
    """Configuration for DLP dataset"""
    max_length: int = 1024
    doc_labels: List[str] = None
    
    def __post_init__(self):
        if self.doc_labels is None:
            self.doc_labels = ["sensitivity", "exposure", "context", "obfuscation"]


def create_dlp_dataloader(
    data_path: str,
    tokenizer: Any,
    dataset_config: DLPDatasetConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DLP dataloader"""
    
    # Use legacy dataset implementation for compatibility
    dataset = LegacyDLPDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=dataset_config.max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )