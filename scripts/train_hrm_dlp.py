"""
HRM-DLP Training Script

Implements the HRM training methodology adapted for DLP tasks:
- 1-step gradient training with carry state management
- Multi-task loss (document + span + ACT)
- Proper AdamATan2 optimizer setup
- Mixed precision training
- ACT evaluation loop
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse
from dataclasses import dataclass
import yaml

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf

try:
    from adam_atan2 import AdamATan2
    HAS_ADAM_ATAN2 = True
except ImportError:
    HAS_ADAM_ATAN2 = False
    print("Warning: adam_atan2 not found, falling back to AdamW")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dlp.dataset import DLPDataset, DLPDatasetConfig
from src.dlp.tokenizer import DLPTokenizer, SimpleTokenizer
from src.dlp.hrm_model import create_hrm_dlp_model, HRMDLPConfig
from src.dlp.hrm_losses import HRMDLPLossWrapper


@dataclass
class HRMDLPTrainingConfig:
    """Training configuration for HRM-DLP."""
    
    # Data
    train_path: str
    val_path: str
    max_length: int = 1024
    tokenizer_path: str = None
    
    # Model architecture
    vocab_size: int = 16000
    hidden_size: int = 384
    num_heads: int = 6
    H_layers: int = 4
    L_layers: int = 4
    H_cycles: int = 2
    L_cycles: int = 2
    expansion: float = 4.0
    
    # ACT settings
    use_act: bool = True
    act_max_steps: int = 4
    act_exploration_prob: float = 0.0
    
    # Training
    epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Learning rate scheduling
    lr_warmup_steps: int = 3000
    lr_min_ratio: float = 0.1
    
    # Loss weights
    doc_loss_weight: float = 1.0
    span_loss_weight: float = 1.0
    act_ponder_weight: float = 0.1
    act_q_weight: float = 0.5
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    # Logging and checkpointing
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    
    # Experiment tracking
    project_name: str = "hrm-dlp-training"
    run_name: str = None
    
    # Output
    checkpoint_dir: str = "checkpoints/hrm_dlp_training"
    
    # System
    seed: int = 42


def create_tokenizer(config: HRMDLPTrainingConfig) -> Any:
    """Create tokenizer for the training."""
    if config.tokenizer_path and os.path.exists(config.tokenizer_path):
        print(f"Loading tokenizer from {config.tokenizer_path}")
        return DLPTokenizer(config.tokenizer_path)
    else:
        print("Using simple fallback tokenizer")
        return SimpleTokenizer(vocab_size=config.vocab_size)


def create_dataset(config: HRMDLPTrainingConfig, split: str, tokenizer: Any) -> DLPDataset:
    """Create dataset for training or validation."""
    
    if split == 'train':
        data_path = config.train_path
    elif split == 'val':
        data_path = config.val_path
    else:
        raise ValueError(f"Unknown split: {split}")
    
    dataset_config = DLPDatasetConfig(
        max_length=config.max_length,
        doc_labels=["sensitivity", "exposure", "context", "obfuscation"]
    )
    
    return DLPDataset(data_path, tokenizer, dataset_config)


def create_model(config: HRMDLPTrainingConfig, vocab_size: int) -> HRMDLPLossWrapper:
    """Create HRM-DLP model with loss wrapper."""
    
    model_config = HRMDLPConfig(
        vocab_size=vocab_size,
        seq_len=config.max_length,
        hidden_size=config.hidden_size,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        num_heads=config.num_heads,
        expansion=config.expansion,
        use_act=config.use_act,
        act_max_steps=config.act_max_steps,
        act_exploration_prob=config.act_exploration_prob,
        forward_dtype=config.amp_dtype
    )
    
    # Create model
    base_model = create_hrm_dlp_model(model_config.__dict__)
    
    # Wrap with loss computation
    model = HRMDLPLossWrapper(
        base_model,
        doc_loss_weight=config.doc_loss_weight,
        span_loss_weight=config.span_loss_weight,
        act_ponder_weight=config.act_ponder_weight,
        act_q_weight=config.act_q_weight
    )
    
    return model


def create_optimizer(model: torch.nn.Module, config: HRMDLPTrainingConfig) -> torch.optim.Optimizer:
    """Create AdamATan2 optimizer following HRM methodology."""
    
    # Use AdamATan2 as in original HRM
    optimizer = AdamATan2(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    return optimizer


def cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_ratio: float = 0.0
) -> float:
    """Cosine learning rate schedule with warmup."""
    
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def train_step(
    model: HRMDLPLossWrapper,
    carry: Any,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: HRMDLPTrainingConfig,
    step: int,
    total_steps: int
) -> Tuple[Any, Dict[str, float]]:
    """
    Perform a single training step with HRM 1-step gradient methodology.
    """
    
    # Move batch to GPU
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Initialize carry if needed
    if carry is None:
        carry = model.initial_carry(batch)
    
    # Forward pass with mixed precision
    if config.use_amp:
        with torch.cuda.amp.autocast(dtype=getattr(torch, config.amp_dtype)):
            carry, loss, metrics, _, all_halted = model(carry=carry, batch=batch, return_keys=[])
    else:
        carry, loss, metrics, _, all_halted = model(carry=carry, batch=batch, return_keys=[])
    
    # Backward pass
    if config.use_amp:
        scaler.scale(loss / config.gradient_accumulation_steps).backward()
    else:
        (loss / config.gradient_accumulation_steps).backward()
    
    # Update learning rate
    lr = cosine_schedule_with_warmup(
        step, config.lr_warmup_steps, total_steps, config.learning_rate, config.lr_min_ratio
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Optimizer step every gradient_accumulation_steps
    if (step + 1) % config.gradient_accumulation_steps == 0:
        if config.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Convert metrics to CPU and add learning rate
    cpu_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            cpu_metrics[f"train/{k}"] = v.cpu().item()
        else:
            cpu_metrics[f"train/{k}"] = v
    
    cpu_metrics["train/lr"] = lr
    cpu_metrics["train/loss_scaled"] = loss.cpu().item()
    
    return carry, cpu_metrics


def evaluate_model(
    model: HRMDLPLossWrapper,
    val_loader: DataLoader,
    config: HRMDLPTrainingConfig
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Initialize carry
            carry = model.initial_carry(batch)
            
            # ACT evaluation loop - keep running until all sequences halt
            while True:
                if config.use_amp:
                    with torch.cuda.amp.autocast(dtype=getattr(torch, config.amp_dtype)):
                        carry, loss, metrics, _, all_halted = model(carry=carry, batch=batch)
                else:
                    carry, loss, metrics, _, all_halted = model(carry=carry, batch=batch)
                
                if all_halted:
                    break
            
            # Collect metrics
            batch_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    batch_metrics[k] = v.cpu().item()
                else:
                    batch_metrics[k] = v
            
            batch_metrics["total_loss"] = loss.cpu().item()
            all_metrics.append(batch_metrics)
    
    model.train()
    
    # Average metrics
    if not all_metrics:
        return {}
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[f"val/{key}"] = sum(values) / len(values)
    
    return avg_metrics


def train_hrm_dlp(config: HRMDLPTrainingConfig):
    """Main training function."""
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Create tokenizer
    tokenizer = create_tokenizer(config)
    vocab_size = getattr(tokenizer, 'vocab_size', config.vocab_size)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(config, 'train', tokenizer)
    val_dataset = create_dataset(config, 'val', tokenizer)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.per_device_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.per_device_batch_size, shuffle=False)
    
    # Create model
    print("Creating HRM-DLP model...")
    model = create_model(config, vocab_size)
    model.cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    # Calculate total steps
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    
    print(f"Total training steps: {total_steps}")
    
    # Initialize wandb
    if config.run_name is None:
        config.run_name = f"hrm-dlp_{int(time.time())}_bs{config.per_device_batch_size * config.gradient_accumulation_steps}_lr{config.learning_rate:.0e}"
    
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.__dict__
    )
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    model.train()
    carry = None
    step = 0
    
    print("Starting training...")
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            carry, metrics = train_step(
                model, carry, batch, optimizer, scaler, config, step, total_steps
            )
            
            epoch_metrics.append(metrics)
            
            # Logging
            if step % config.log_every_n_steps == 0:
                wandb.log(metrics, step=step)
                
                # Print progress
                if metrics:
                    loss_val = metrics.get('train/loss/total', metrics.get('train/loss_scaled', 0))
                    lr_val = metrics.get('train/lr', 0)
                    print(f"Step {step}: Loss = {loss_val:.4f}, LR = {lr_val:.2e}")
            
            # Evaluation
            if step > 0 and step % config.eval_every_n_steps == 0:
                print(f"Evaluating at step {step}...")
                val_metrics = evaluate_model(model, val_loader, config)
                
                if val_metrics:
                    wandb.log(val_metrics, step=step)
                    val_loss = val_metrics.get('val/total_loss', 0)
                    val_acc = val_metrics.get('val/doc_accuracy', 0)
                    print(f"Validation - Loss: {val_loss:.4f}, Doc Accuracy: {val_acc:.4f}")
            
            # Checkpointing
            if step > 0 and step % config.save_every_n_steps == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__,
                    'carry': carry
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            step += 1
        
        # End of epoch evaluation
        print(f"Evaluating end of epoch {epoch + 1}...")
        val_metrics = evaluate_model(model, val_loader, config)
        
        if val_metrics:
            wandb.log(val_metrics, step=step)
            val_loss = val_metrics.get('val/total_loss', 0)
            val_acc = val_metrics.get('val/doc_accuracy', 0)
            print(f"End of Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_checkpoint.pt")
    torch.save({
        'step': step,
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'carry': carry
    }, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")
    
    wandb.finish()
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train HRM-DLP model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--train_path", type=str, help="Path to training data")
    parser.add_argument("--val_path", type=str, help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/hrm_dlp_training", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = HRMDLPTrainingConfig(**config_dict)
    else:
        # Use command line arguments
        config = HRMDLPTrainingConfig(
            train_path=args.train_path or "data/dlp_ag/train_split.jsonl",
            val_path=args.val_path or "data/dlp_ag/val_split.jsonl",
            epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir
        )
    
    print("Starting HRM-DLP training with config:")
    print(yaml.dump(config.__dict__, default_flow_style=False))
    
    train_hrm_dlp(config)


if __name__ == "__main__":
    main()