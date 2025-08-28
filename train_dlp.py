#!/usr/bin/env python3
"""
HRM-DLP Training Script

Trains the Hierarchical Reasoning Model for Data Loss Prevention tasks.
Uses the proper DLP implementation from src/dlp/ with ACT, hierarchical reasoning,
and DLP-specific task heads.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.dlp.hrm_model import HRMDLP, HRMDLPConfig, HRMDLPOutput
from src.dlp.dataset import DLPDataset, DLPDatasetConfig, create_dataloaders
from src.dlp.tokenizer import DLPTokenizer, SimpleTokenizer, TokenizerConfig, create_tokenizer
from src.dlp.act import ACTLoss
from src.dlp.dsl import NUM_BIO_TAGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DLPTrainingConfig:
    """Training configuration for HRM-DLP."""
    
    def __init__(self, **kwargs):
        # Data paths
        self.train_path: str = kwargs.get('train_path', 'data/hrm_dlp_final/train.jsonl')
        self.val_path: str = kwargs.get('val_path', 'data/hrm_dlp_final/val.jsonl')
        self.test_path: str = kwargs.get('test_path', 'data/hrm_dlp_final/test.jsonl')
        
        # Model config
        self.vocab_size: int = kwargs.get('vocab_size', 16000)
        self.seq_len: int = kwargs.get('seq_len', 1024)
        self.hidden_size: int = kwargs.get('hidden_size', 384)
        
        # HRM architecture (docs specify 8 fast + 2 slow layers)
        self.H_layers: int = kwargs.get('H_layers', 2)
        self.L_layers: int = kwargs.get('L_layers', 8)
        self.H_cycles: int = kwargs.get('H_cycles', 2)
        self.L_cycles: int = kwargs.get('L_cycles', 2)
        
        # Transformer config
        self.num_heads: int = kwargs.get('num_heads', 6)
        self.expansion: float = kwargs.get('expansion', 4.0)
        self.rms_norm_eps: float = kwargs.get('rms_norm_eps', 1e-5)
        self.rope_theta: float = kwargs.get('rope_theta', 10000.0)
        
        # DLP task heads
        self.num_doc_scores: int = kwargs.get('num_doc_scores', 4)  # sensitivity, exposure, context, obfuscation
        self.num_span_tags: int = NUM_BIO_TAGS
        self.memory_vec_dim: int = kwargs.get('memory_vec_dim', 256)
        
        # ACT configuration
        self.use_act: bool = kwargs.get('use_act', True)
        self.act_max_steps: int = kwargs.get('act_max_steps', 4)
        self.act_exploration_prob: float = kwargs.get('act_exploration_prob', 0.1)
        
        # Segment processing
        self.segment_size: int = kwargs.get('segment_size', 64)
        
        # Training hyperparameters
        self.batch_size: int = kwargs.get('batch_size', 16)
        self.learning_rate: float = kwargs.get('learning_rate', 1e-4)
        self.weight_decay: float = kwargs.get('weight_decay', 0.01)
        self.num_epochs: int = kwargs.get('num_epochs', 10)
        self.warmup_steps: int = kwargs.get('warmup_steps', 1000)
        self.grad_clip_norm: float = kwargs.get('grad_clip_norm', 1.0)
        
        # Dataset config
        self.max_length: int = kwargs.get('max_length', 1024)
        self.doc_labels: list = kwargs.get('doc_labels', ["sensitivity", "exposure", "context", "obfuscation"])
        self.pad_token_id: int = kwargs.get('pad_token_id', 0)
        self.ignore_label_id: int = kwargs.get('ignore_label_id', -100)
        
        # Training setup
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.forward_dtype: str = kwargs.get('forward_dtype', 'bfloat16')
        self.num_workers: int = kwargs.get('num_workers', 4)
        self.save_every_n_steps: int = kwargs.get('save_every_n_steps', 1000)
        self.eval_every_n_steps: int = kwargs.get('eval_every_n_steps', 500)
        
        # Logging
        self.wandb_project: str = kwargs.get('wandb_project', 'hrm-dlp')
        self.wandb_run_name: Optional[str] = kwargs.get('wandb_run_name', None)
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', 'checkpoints/hrm_dlp')
        
        # Loss weights (as per docs: BCE(doc) + CE(BIO) + 0.3*mask-denoise + 0.2*section-shuffle)
        self.doc_loss_weight: float = kwargs.get('doc_loss_weight', 1.0)
        self.span_loss_weight: float = kwargs.get('span_loss_weight', 1.0)
        self.act_loss_weight: float = kwargs.get('act_loss_weight', 0.1)
        self.mask_denoise_weight: float = kwargs.get('mask_denoise_weight', 0.3)
        self.section_shuffle_weight: float = kwargs.get('section_shuffle_weight', 0.2)
        
        # Auxiliary loss parameters
        self.mask_denoise_prob: float = kwargs.get('mask_denoise_prob', 0.15)  # 15% as per docs
        self.section_shuffle_prob: float = kwargs.get('section_shuffle_prob', 0.10)  # 10% as per docs


class DLPLossComputer:
    """Computes multi-task losses for DLP training."""
    
    def __init__(self, config: DLPTrainingConfig):
        self.config = config
        self.doc_criterion = nn.BCEWithLogitsLoss()  # Multi-label document classification
        self.span_criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label_id)
        
        # Auxiliary loss criteria
        self.mask_denoise_criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label_id)
        self.section_shuffle_criterion = nn.BCEWithLogitsLoss()
        
        if config.use_act:
            self.act_loss = ACTLoss(ponder_weight=0.1)
    
    def compute_loss(self, output: HRMDLPOutput, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all DLP losses."""
        losses = {}
        
        # Document classification loss
        doc_loss = self.doc_criterion(output.doc_logits, batch['doc_labels'])
        losses['doc_loss'] = doc_loss
        
        # Span tagging loss
        span_logits_flat = output.span_logits.view(-1, output.span_logits.size(-1))
        span_labels_flat = batch['bio_labels'].view(-1)
        span_loss = self.span_criterion(span_logits_flat, span_labels_flat)
        losses['span_loss'] = span_loss
        
        # ACT loss if enabled
        if self.config.use_act and hasattr(output, 'q_halt_logits'):
            act_loss_value = self.act_loss(
                output.q_halt_logits,
                output.q_continue_logits,
                output.steps,
                target_q_continue=output.target_q_continue
            )
            losses['act_loss'] = act_loss_value
        
        # Auxiliary losses (as per docs: 0.3*mask-denoise + 0.2*section-shuffle)
        if 'masked_input_ids' in batch and 'mask_targets' in batch:
            # Mask-denoise loss: predict masked tokens
            mask_denoise_loss = self._compute_mask_denoise_loss(output, batch)
            losses['mask_denoise_loss'] = mask_denoise_loss
        else:
            mask_denoise_loss = torch.tensor(0.0, device=doc_loss.device)
            
        if 'section_shuffle_labels' in batch:
            # Section shuffle detection loss: binary classification of shuffled segments
            section_shuffle_loss = self._compute_section_shuffle_loss(output, batch)
            losses['section_shuffle_loss'] = section_shuffle_loss
        else:
            section_shuffle_loss = torch.tensor(0.0, device=doc_loss.device)
        
        # Total loss (as per docs formula)
        total_loss = (
            self.config.doc_loss_weight * doc_loss +
            self.config.span_loss_weight * span_loss +
            self.config.mask_denoise_weight * mask_denoise_loss +
            self.config.section_shuffle_weight * section_shuffle_loss
        )
        
        if self.config.use_act and 'act_loss' in losses:
            total_loss += self.config.act_loss_weight * losses['act_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_mask_denoise_loss(self, output: HRMDLPOutput, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute mask-denoise auxiliary loss.
        
        The model should predict the original tokens at masked positions.
        This requires an additional head that predicts vocabulary tokens.
        """
        # For now, use span logits as a proxy (this would need a separate vocab head in practice)
        # This is a simplified implementation
        if 'mask_positions' in batch and 'mask_targets' in batch:
            mask_positions = batch['mask_positions']  # [batch, num_masks]
            mask_targets = batch['mask_targets']      # [batch, num_masks]
            
            # Extract logits at masked positions (simplified)
            # In practice, you'd need a separate vocabulary prediction head
            batch_size, seq_len, vocab_size = output.span_logits.shape
            
            # Use document logits as proxy for token-level prediction (simplified)
            masked_logits = output.doc_logits.mean(dim=1, keepdim=True).expand(batch_size, mask_targets.size(1))
            
            # Convert targets to appropriate range
            mask_targets_clamped = torch.clamp(mask_targets, 0, output.doc_logits.size(-1) - 1)
            
            return self.mask_denoise_criterion(masked_logits.unsqueeze(-1).expand(-1, -1, 4), 
                                             mask_targets_clamped % 4)
        
        return torch.tensor(0.0, device=output.doc_logits.device)
    
    def _compute_section_shuffle_loss(self, output: HRMDLPOutput, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute section shuffle detection loss.
        
        Binary classification: predict whether document sections were shuffled.
        """
        if 'section_shuffle_labels' in batch:
            # Use CLS token (first token) representation for binary classification
            shuffle_labels = batch['section_shuffle_labels'].float()  # [batch]
            
            # Use first document score as shuffle detection proxy
            shuffle_logits = output.doc_logits[:, 0]  # [batch]
            
            return self.section_shuffle_criterion(shuffle_logits, shuffle_labels)
        
        return torch.tensor(0.0, device=output.doc_logits.device)


def create_model_and_tokenizer(config: DLPTrainingConfig):
    """Create HRM-DLP model and tokenizer."""
    
    # Create tokenizer - try SentencePiece first, fall back to SimpleTokenizer
    try:
        # Check if we have training data to create a tokenizer
        tokenizer_model_path = f"{config.checkpoint_dir}/tokenizer.model"
        if os.path.exists(tokenizer_model_path):
            # Load existing tokenizer
            tokenizer_config = TokenizerConfig(vocab_size=config.vocab_size)
            tokenizer = DLPTokenizer(tokenizer_model_path, tokenizer_config)
            logger.info(f"Loaded existing tokenizer from {tokenizer_model_path}")
        else:
            # Create new tokenizer from training data
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            tokenizer_config = TokenizerConfig(vocab_size=config.vocab_size)
            
            try:
                tokenizer = create_tokenizer(
                    [config.train_path, config.val_path],
                    f"{config.checkpoint_dir}/tokenizer", 
                    tokenizer_config
                )
                logger.info("Created new SentencePiece tokenizer from training data")
            except Exception as e:
                logger.warning(f"Failed to create SentencePiece tokenizer: {e}")
                logger.info("Falling back to SimpleTokenizer")
                tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
                
    except ImportError:
        logger.warning("SentencePiece not available, using SimpleTokenizer")
        tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create model config
    model_config = HRMDLPConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        hidden_size=config.hidden_size,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        num_heads=config.num_heads,
        expansion=config.expansion,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        num_doc_scores=config.num_doc_scores,
        num_span_tags=config.num_span_tags,
        memory_vec_dim=config.memory_vec_dim,
        use_act=config.use_act,
        act_max_steps=config.act_max_steps,
        act_exploration_prob=config.act_exploration_prob,
        segment_size=config.segment_size,
        forward_dtype=config.forward_dtype
    )
    
    # Create model
    model = HRMDLP(model_config)
    
    return model, tokenizer


def create_optimizer_and_scheduler(model: nn.Module, config: DLPTrainingConfig, total_steps: int):
    """Create optimizer and learning rate scheduler."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_step(model: HRMDLP, batch: Dict[str, torch.Tensor], loss_computer: DLPLossComputer, 
               optimizer: torch.optim.Optimizer, config: DLPTrainingConfig) -> Dict[str, float]:
    """Single training step with ACT loop."""
    
    model.train()
    
    # Initialize carry
    carry = model.initial_carry(batch)
    
    # ACT loop - continue until all sequences halt
    step_outputs = []
    max_act_steps = config.act_max_steps if config.use_act else 1
    
    for act_step in range(max_act_steps):
        carry, output, all_halted = model(carry, batch)
        step_outputs.append(output)
        
        if all_halted:
            break
    
    # Use output from final step
    final_output = step_outputs[-1]
    
    # Compute losses
    losses = loss_computer.compute_loss(final_output, batch)
    
    # Backward pass
    total_loss = losses['total_loss']
    total_loss.backward()
    
    # Gradient clipping
    if config.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Return metrics
    metrics = {k: v.item() for k, v in losses.items()}
    metrics['act_steps'] = len(step_outputs)
    metrics['lr'] = optimizer.param_groups[0]['lr']
    
    return metrics


def evaluate(model: HRMDLP, dataloader: DataLoader, loss_computer: DLPLossComputer, 
             config: DLPTrainingConfig) -> Dict[str, float]:
    """Evaluate model on validation set."""
    
    model.eval()
    total_metrics = {}
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Initialize carry
            carry = model.initial_carry(batch)
            
            # ACT loop
            max_act_steps = config.act_max_steps if config.use_act else 1
            for act_step in range(max_act_steps):
                carry, output, all_halted = model(carry, batch)
                if all_halted:
                    break
            
            # Compute losses
            losses = loss_computer.compute_loss(output, batch)
            
            # Accumulate metrics
            for k, v in losses.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v.item()
            
            total_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / total_batches for k, v in total_metrics.items()}
    return avg_metrics


def save_checkpoint(model: HRMDLP, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler.LambdaLR, step: int,
                   config: DLPTrainingConfig, metrics: Dict[str, float]):
    """Save model checkpoint."""
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'config': config.__dict__,
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train HRM-DLP model")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--train_path", type=str, help="Path to training data")
    parser.add_argument("--val_path", type=str, help="Path to validation data") 
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load config
    config_dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    
    # Override with command line args
    if args.train_path:
        config_dict['train_path'] = args.train_path
    if args.val_path:
        config_dict['val_path'] = args.val_path
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config_dict['num_epochs'] = args.num_epochs
        
    config = DLPTrainingConfig(**config_dict)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )
    
    logger.info("Creating model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config)
    model = model.to(config.device)
    
    logger.info("Creating datasets and dataloaders...")
    dataset_config = DLPDatasetConfig(
        max_length=config.max_length,
        doc_labels=config.doc_labels,
        pad_token_id=config.pad_token_id,
        ignore_label_id=config.ignore_label_id
    )
    
    train_loader, val_loader = create_dataloaders(
        config.train_path,
        config.val_path,
        tokenizer,
        dataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Estimate total steps
    total_steps = len(train_loader) * config.num_epochs
    
    logger.info("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
    
    logger.info("Creating loss computer...")
    loss_computer = DLPLossComputer(config)
    
    logger.info(f"Starting training for {config.num_epochs} epochs...")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Training on {len(train_loader)} batches, validation on {len(val_loader)} batches")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Training loop
        model.train()
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch_dict in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch_dict.items()}
            
            # Training step
            step_metrics = train_step(model, batch, loss_computer, optimizer, config)
            scheduler.step()
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['total_loss']:.4f}",
                'doc_loss': f"{step_metrics['doc_loss']:.4f}",
                'span_loss': f"{step_metrics['span_loss']:.4f}",
                'lr': f"{step_metrics['lr']:.2e}"
            })
            
            global_step += 1
            
            # Log to wandb
            if args.use_wandb:
                wandb.log(step_metrics, step=global_step)
            
            # Evaluation
            if global_step % config.eval_every_n_steps == 0:
                logger.info("Running evaluation...")
                val_metrics = evaluate(model, val_loader, loss_computer, config)
                
                # Add val prefix to metrics
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                
                logger.info(f"Validation metrics: {val_metrics}")
                
                if args.use_wandb:
                    wandb.log(val_metrics, step=global_step)
                
                # Save checkpoint if best validation loss
                if val_metrics['val_total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_total_loss']
                    save_checkpoint(model, optimizer, scheduler, global_step, config, val_metrics)
            
            # Save checkpoint periodically
            if global_step % config.save_every_n_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, config, step_metrics)
        
        # Log epoch metrics
        avg_epoch_metrics = {f"epoch_{k}": sum(v) / len(v) for k, v in epoch_metrics.items()}
        logger.info(f"Epoch {epoch + 1} metrics: {avg_epoch_metrics}")
        
        if args.use_wandb:
            wandb.log(avg_epoch_metrics, step=global_step)
    
    # Final evaluation and checkpoint
    logger.info("Final evaluation...")
    val_metrics = evaluate(model, val_loader, loss_computer, config)
    val_metrics = {f"final_val_{k}": v for k, v in val_metrics.items()}
    
    logger.info(f"Final validation metrics: {val_metrics}")
    
    if args.use_wandb:
        wandb.log(val_metrics, step=global_step)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, config, val_metrics)
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()