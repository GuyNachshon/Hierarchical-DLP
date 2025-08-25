"""
DLP Trainer

Training loop and utilities for DLP models.
"""

import torch
import wandb
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DLPTrainingConfig:
    """Configuration for DLP training"""
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.02
    global_batch_size: int = 96
    gradient_accumulation_steps: int = 6
    lr_warmup_steps: int = 3000
    lr_min_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_metric: str = "eval/total_loss"
    early_stopping_mode: str = "min"
    project_name: str = "hrm-dlp-training"
    run_name: Optional[str] = None
    use_wandb: bool = True


class DLPTrainer:
    """Trainer for DLP models"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader,
        val_dataloader,
        loss_fn,
        config: DLPTrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.__dict__
            )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Training phase
            train_stats = self._train_epoch()
            
            # Evaluation phase
            eval_stats = self._eval_epoch()
            
            # Check for early stopping
            current_eval_loss = eval_stats.get("eval/total_loss", float('inf'))
            if current_eval_loss < self.best_eval_loss:
                self.best_eval_loss = current_eval_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping after {self.patience_counter} epochs without improvement")
                break
            
            # Log epoch stats
            epoch_stats = {**train_stats, **eval_stats}
            if self.config.use_wandb:
                wandb.log(epoch_stats, step=self.global_step)
            
            print(f"Epoch {epoch + 1} - Train Loss: {train_stats.get('train/total_loss', 0):.4f}, "
                  f"Eval Loss: {eval_stats.get('eval/total_loss', 0):.4f}")
        
        # Save final checkpoint
        self._save_checkpoint("final_model.pt")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return {
            "final_train_loss": train_stats.get("train/total_loss", 0),
            "final_eval_loss": eval_stats.get("eval/total_loss", 0),
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                output, _ = self.model(batch["input_ids"])
                loss_dict = self.loss_fn(
                    doc_logits=output.doc_logits,
                    span_logits=output.span_logits,
                    hidden_states=output.hidden_states,
                    doc_labels=batch["doc_labels"],
                    span_labels=batch["span_labels"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None)
                )
                loss = loss_dict["total_loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update parameters
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    if self.config.use_wandb:
                        wandb.log({
                            "train/total_loss": loss_dict["total_loss"].item(),
                            "train/doc_loss": loss_dict.get("doc_loss", 0),
                            "train/span_loss": loss_dict.get("span_loss", 0),
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train/step": self.global_step
                        }, step=self.global_step)
            
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_dataloader)} - Loss: {loss_dict['total_loss'].item():.4f}")
        
        return {
            "train/total_loss": total_loss / num_batches if num_batches > 0 else 0
        }
    
    def _eval_epoch(self) -> Dict[str, float]:
        """Evaluate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                output, _ = self.model(batch["input_ids"])
                loss_dict = self.loss_fn(
                    doc_logits=output.doc_logits,
                    span_logits=output.span_logits,
                    hidden_states=output.hidden_states,
                    doc_labels=batch["doc_labels"],
                    span_labels=batch["span_labels"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    compute_auxiliary=False
                )
                
                total_loss += loss_dict["total_loss"].item()
                num_batches += 1
        
        return {
            "eval/total_loss": total_loss / num_batches if num_batches > 0 else 0
        }
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Saved checkpoint: {checkpoint_dir / filename}")