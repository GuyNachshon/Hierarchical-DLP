"""
Training Utilities

Common functions for model training setup, checkpointing, and optimization.
"""

import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional


def setup_training(local_rank: int = 0, world_size: int = 1) -> Dict[str, Any]:
    """
    Setup distributed training environment
    
    Returns:
        Dictionary with device and distributed training info
    """
    if world_size > 1:
        # Initialize distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return {
        "device": device,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_distributed": world_size > 1
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, step: int, loss: float, checkpoint_dir: str,
                   filename: Optional[str] = None) -> str:
    """
    Save model checkpoint
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    checkpoint_path = checkpoint_dir / filename
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Returns:
        Dictionary with loaded checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf"))
    }


def get_optimizer(model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    Create optimizer with standard settings
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_lr_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, 
                    warmup_steps: int = 1000, min_lr_ratio: float = 0.1):
    """
    Create learning rate scheduler
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Simple cosine annealing for now
    # TODO: Add more sophisticated scheduling with warmup
    return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio)