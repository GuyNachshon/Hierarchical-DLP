"""Training script for HRM-DLP model

Adapted from original HRM pretraining for DLP-specific multi-task objectives.
"""

from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import json
import yaml
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf
from adam_atan2_pytorch import AdamAtan2

import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dlp.model import HRMDLPModel, DLPModelConfig
from src.dlp.dataset import DLPDataset, DLPDatasetConfig
from src.dlp.losses import DLPMultiTaskLoss, DLPLossConfig
from src.dlp.tokenizer import DLPTokenizer


class SimpleTokenizer:
    """Simple character-level tokenizer fallback"""
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text: str, add_bos=False, add_eos=False):
        # Simple character encoding (ASCII)
        tokens = [min(ord(c), self.vocab_size-1) for c in text]
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
        return tokens
    
    def decode(self, token_ids):
        return ''.join([chr(min(max(t, 32), 126)) for t in token_ids if t > 3])


class DLPArchConfig(pydantic.BaseModel):
    """Architecture configuration for DLP model"""
    model_config = pydantic.ConfigDict(extra='allow')
    
    # Model architecture
    hidden_size: int = 384
    num_heads: int = 6
    H_layers: int = 4
    L_layers: int = 4
    H_cycles: int = 2
    L_cycles: int = 2
    expansion: float = 4.0
    pos_encodings: str = "rope"
    
    # DLP-specific
    num_doc_labels: int = 4
    num_bio_tags: int = 21
    memory_dim: int = 256
    use_fusion_gates: bool = True
    
    # Training behavior
    use_act: bool = False  # Disable ACT for deterministic inference
    forward_dtype: str = "bfloat16"


class DLPTrainConfig(pydantic.BaseModel):
    """Training configuration for DLP model"""
    # Data
    data_path: str
    train_path: str
    val_path: str
    tokenizer_path: Optional[str] = None
    max_length: int = 1024
    
    # Model architecture
    arch: DLPArchConfig
    
    # Training hyperparameters
    global_batch_size: int = 768
    epochs: int = 2
    lr: float = 3e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 3000
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Loss weights
    doc_loss_weight: float = 1.0
    span_loss_weight: float = 1.0
    mask_denoise_weight: float = 0.3
    section_shuffle_weight: float = 0.2
    label_smoothing: float = 0.05
    
    # Evaluation
    eval_interval: int = 1000
    checkpoint_every_eval: bool = True
    eval_save_outputs: List[str] = []
    
    # Experiment tracking
    project_name: str = "hrm-dlp"
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    # System
    seed: int = 42
    num_workers: int = 4


@dataclass
class DLPTrainState:
    """Training state for DLP model"""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    tokenizer: Any
    
    step: int
    epoch: int
    best_val_loss: float
    
    
def create_dlp_dataloaders(config: DLPTrainConfig, tokenizer, train_path: str, val_path: str) -> tuple:
    """Create training and validation dataloaders"""
    dataset_config = DLPDatasetConfig(
        max_length=config.max_length,
        doc_labels=["sensitivity", "exposure", "context", "obfuscation"]
    )
    
    # Create datasets with provided paths
    train_dataset = DLPDataset(train_path, tokenizer, dataset_config)
    val_dataset = DLPDataset(val_path, tokenizer, dataset_config)
    
    # Check if datasets are empty and handle gracefully
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty. Check that {train_path} exists and contains valid data.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty. Check that {val_path} exists and contains valid data.")
    
    # Create dataloaders  
    batch_size = config.global_batch_size // dist.get_world_size() if dist.is_initialized() else config.global_batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader


def create_model_and_loss(config: DLPTrainConfig, vocab_size: int) -> tuple:
    """Create model and loss function"""
    # Create model config
    model_config_dict = {
        "batch_size": config.global_batch_size,
        "seq_len": config.max_length,
        "vocab_size": vocab_size,
        **config.arch.model_dump()
    }
    
    # Create model
    model_config = DLPModelConfig(**model_config_dict)
    model = HRMDLPModel(model_config)
    
    # Create loss function
    loss_config = DLPLossConfig(
        doc_loss_weight=config.doc_loss_weight,
        span_loss_weight=config.span_loss_weight,
        mask_denoise_weight=config.mask_denoise_weight,
        section_shuffle_weight=config.section_shuffle_weight,
        label_smoothing=config.label_smoothing
    )
    
    loss_fn = DLPMultiTaskLoss(loss_config, vocab_size, config.arch.hidden_size)
    
    return model, loss_fn


def create_optimizer(model: nn.Module, config: DLPTrainConfig) -> torch.optim.Optimizer:
    """Create optimizer with parameter groups"""
    # Separate parameters for different learning rates if needed
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": config.lr,
            "weight_decay": config.weight_decay
        }
    ]
    
    optimizer = AdamAtan2(
        model.parameters(),
        lr=config.lr,  # Will be updated by scheduler
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: DLPTrainConfig, total_steps: int):
    """Create learning rate scheduler"""
    def lr_lambda(step):
        if step < config.lr_warmup_steps:
            # Warmup
            return step / config.lr_warmup_steps
        else:
            # Cosine decay
            progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
            return config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Any) -> Dict[str, float]:
    """Compute evaluation metrics"""
    metrics = {}
    
    # Document classification metrics
    doc_logits = outputs['doc_logits']
    doc_labels = batch.doc_labels
    
    # Convert to probabilities
    doc_probs = torch.sigmoid(doc_logits)
    
    # Binary classification metrics for each head
    doc_names = ["sensitivity", "exposure", "context", "obfuscation"]
    for i, name in enumerate(doc_names):
        pred = (doc_probs[:, i] > 0.5).float()
        target = doc_labels[:, i]
        
        # Accuracy
        acc = (pred == target).float().mean().item()
        metrics[f"doc_{name}_acc"] = acc
        
        # Precision/Recall (avoid division by zero)
        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f"doc_{name}_precision"] = precision
        metrics[f"doc_{name}_recall"] = recall
        metrics[f"doc_{name}_f1"] = f1
    
    # Span tagging metrics
    span_logits = outputs['span_logits']
    span_labels = batch.bio_labels
    
    # Only compute on non-ignored positions
    valid_mask = span_labels != -100
    if valid_mask.sum() > 0:
        span_preds = span_logits.argmax(dim=-1)
        span_acc = ((span_preds == span_labels) * valid_mask).sum().float() / valid_mask.sum().float()
        metrics["span_accuracy"] = span_acc.item()
    else:
        metrics["span_accuracy"] = 0.0
    
    return metrics


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    step: int
) -> Dict[str, float]:
    """Single training step"""
    model.train()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(batch.input_ids, batch.attention_mask)
        
        losses = loss_fn(
            doc_logits=outputs.doc_logits,
            span_logits=outputs.span_logits,
            hidden_states=outputs.hidden_states,
            doc_labels=batch.doc_labels,
            span_labels=batch.bio_labels,
            input_ids=batch.input_ids
        )
        
        total_loss = losses['total_loss']
    
    # Backward pass
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    optimizer.zero_grad(set_to_none=True)
    
    # Convert losses to float for logging
    loss_dict = {k: v.item() for k, v in losses.items()}
    loss_dict['lr'] = scheduler.get_last_lr()[0]
    
    return loss_dict


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    
    total_losses = {}
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            batch.input_ids = batch.input_ids.to(device)
            batch.attention_mask = batch.attention_mask.to(device)
            batch.doc_labels = batch.doc_labels.to(device)
            batch.bio_labels = batch.bio_labels.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(batch.input_ids, batch.attention_mask)
                
                losses = loss_fn(
                    doc_logits=outputs.doc_logits,
                    span_logits=outputs.span_logits,
                    hidden_states=outputs.hidden_states,
                    doc_labels=batch.doc_labels,
                    span_labels=batch.bio_labels,
                    input_ids=batch.input_ids
                )
            
            # Accumulate losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0
                total_losses[k] += v.item()
            
            # Compute metrics
            metrics = compute_metrics(outputs.__dict__, batch)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # Average losses and metrics
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return {**avg_losses, **avg_metrics}


def save_checkpoint(
    train_state: DLPTrainState,
    config: DLPTrainConfig,
    checkpoint_dir: str,
    step: int,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_state_dict": train_state.optimizer.state_dict(),
        "step": step,
        "epoch": train_state.epoch,
        "best_val_loss": train_state.best_val_loss,
        "config": config.model_dump()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        torch.save(checkpoint, best_path)
    
    # Save config for reproducibility
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(), f)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def train(config: DLPTrainConfig):
    """Main training function"""
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Initialize distributed training if available
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
    
    print(f"Using device: {device}, rank: {rank}/{world_size}")
    
    # Create tokenizer
    if config.tokenizer_path and os.path.exists(config.tokenizer_path):
        tokenizer = DLPTokenizer(config.tokenizer_path)
    else:
        print("Warning: Using simple tokenizer. Train a proper tokenizer for better results.")
        tokenizer = SimpleTokenizer(vocab_size=16000)
    
    # Create dataloaders - this will be set by main function
    train_loader, val_loader = create_dlp_dataloaders(config, tokenizer, config.train_path, config.val_path)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 16000
    
    # Create model and loss
    model, loss_fn = create_model_and_loss(config, vocab_size)
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.epochs
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize training state
    train_state = DLPTrainState(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        step=0,
        epoch=0,
        best_val_loss=float('inf')
    )
    
    # Setup experiment tracking
    if rank == 0:
        run_name = config.run_name or coolname.generate_slug(2)
        wandb.init(
            project=config.project_name,
            name=run_name,
            config=config.model_dump()
        )
        
        # Create checkpoint directory
        checkpoint_dir = config.checkpoint_path or f"checkpoints/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)")
    
    for epoch in range(config.epochs):
        train_state.epoch = epoch
        
        # Training epoch
        model.train()
        epoch_losses = {}
        num_batches = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            batch.input_ids = batch.input_ids.to(device)
            batch.attention_mask = batch.attention_mask.to(device)
            batch.doc_labels = batch.doc_labels.to(device)
            batch.bio_labels = batch.bio_labels.to(device)
            
            # Training step
            step_losses = train_step(
                model, loss_fn, batch, optimizer, scheduler, scaler, train_state.step
            )
            
            # Accumulate losses
            for k, v in step_losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v
            
            num_batches += 1
            train_state.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{step_losses['total_loss']:.4f}",
                "lr": f"{step_losses['lr']:.2e}"
            })
            
            # Evaluation
            if train_state.step % config.eval_interval == 0:
                eval_results = evaluate(model, loss_fn, val_loader, device)
                
                if rank == 0:
                    print(f"\nStep {train_state.step} Evaluation:")
                    for k, v in eval_results.items():
                        print(f"  {k}: {v:.4f}")
                    
                    # Log to wandb
                    wandb.log({
                        **{f"train/{k}": v / num_batches for k, v in epoch_losses.items()},
                        **{f"val/{k}": v for k, v in eval_results.items()},
                        "step": train_state.step,
                        "epoch": epoch
                    })
                    
                    # Save checkpoint
                    is_best = eval_results['total_loss'] < train_state.best_val_loss
                    if is_best:
                        train_state.best_val_loss = eval_results['total_loss']
                    
                    if config.checkpoint_every_eval:
                        save_checkpoint(train_state, config, checkpoint_dir, train_state.step, is_best)
                
                # Reset epoch tracking
                epoch_losses = {}
                num_batches = 0
    
    # Final evaluation and checkpoint
    if rank == 0:
        final_eval = evaluate(model, loss_fn, val_loader, device)
        print(f"\nFinal Evaluation Results:")
        for k, v in final_eval.items():
            print(f"  {k}: {v:.4f}")
        
        # Save final checkpoint
        save_checkpoint(train_state, config, checkpoint_dir, train_state.step, is_best=True)
        
        wandb.finish()
    
    print("Training completed!")


@hydra.main(version_base=None, config_path="../config/training", config_name="dlp_training")
def main(cfg: DictConfig):
    """Main entry point"""
    # Convert the structured config to our DLPTrainConfig format
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Determine which data paths to use (with fallback logic)
    train_path = cfg_dict['data']['train_path']
    val_path = cfg_dict['data']['val_path']
    
    # Check if files exist, use fallbacks if needed
    if not os.path.exists(train_path):
        print(f"Primary train path {train_path} not found, trying fallback...")
        fallback_train = cfg_dict['data'].get('fallback_train_path')
        if fallback_train and os.path.exists(fallback_train):
            train_path = fallback_train
            print(f"Using fallback train path: {train_path}")
        else:
            print(f"Warning: No valid training data found!")
    
    if not os.path.exists(val_path):
        print(f"Primary val path {val_path} not found, trying fallback...")
        fallback_val = cfg_dict['data'].get('fallback_val_path') 
        if fallback_val and os.path.exists(fallback_val):
            val_path = fallback_val
            print(f"Using fallback val path: {val_path}")
        else:
            print(f"Warning: No valid validation data found!")
    
    # Map the nested structure to flat structure
    config_dict = {
        # Data config
        'data_path': os.path.dirname(train_path),  # Extract base path from train path
        'train_path': train_path,
        'val_path': val_path,
        'tokenizer_path': cfg_dict['data'].get('tokenizer_path'),
        'max_length': cfg_dict['data']['max_length'],
        
        # Training config
        'epochs': cfg_dict['training']['epochs'],
        'lr': cfg_dict['training']['learning_rate'],
        'lr_min_ratio': cfg_dict['training']['lr_min_ratio'],
        'lr_warmup_steps': cfg_dict['training']['lr_warmup_steps'],
        'weight_decay': cfg_dict['training']['weight_decay'],
        'beta1': 0.9,  # Default
        'beta2': 0.95, # Default
        
        # Batch size calculation
        'global_batch_size': cfg_dict['training']['per_device_batch_size'] * cfg_dict['training']['gradient_accumulation_steps'],
        
        # Evaluation
        'eval_interval': cfg_dict['training']['eval_every_n_steps'],
        'checkpoint_every_eval': True,
        'eval_save_outputs': [],
        
        # Experiment tracking  
        'project_name': cfg_dict['experiment']['project_name'],
        'run_name': cfg_dict['experiment'].get('run_name'),
        'checkpoint_path': cfg_dict['output']['checkpoint_dir'],
        
        # System
        'seed': 42,  # Default
        'num_workers': 4,  # Default
        
        # Architecture (use defaults from DLPArchConfig)
        'arch': DLPArchConfig(),
        
        # Loss weights (use defaults)
        'doc_loss_weight': 1.0,
        'span_loss_weight': 1.0, 
        'mask_denoise_weight': 0.3,
        'section_shuffle_weight': 0.2,
        'label_smoothing': 0.05,
    }
    
    config = DLPTrainConfig(**config_dict)
    train(config)


if __name__ == "__main__":
    main()