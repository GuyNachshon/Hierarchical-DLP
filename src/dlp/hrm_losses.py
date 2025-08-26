"""
HRM-DLP Loss Functions

Combines multi-task DLP losses with ACT (Adaptive Compute Time) losses:
- Document classification (BCE)
- Span tagging (Cross-entropy) 
- ACT ponder cost and Q-learning losses
- Auxiliary objectives (mask-denoise, section-shuffle)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn

from .act import ACTLoss
from .hrm_model import HRMDLPOutput


class HRMDLPMultiTaskLoss(nn.Module):
    """
    Multi-task loss for HRM-DLP combining:
    1. Document classification loss (BCE)
    2. Span tagging loss (Cross-entropy)
    3. ACT losses (ponder + Q-learning)
    4. Auxiliary objectives
    """
    
    def __init__(
        self,
        doc_loss_weight: float = 1.0,
        span_loss_weight: float = 1.0,
        act_ponder_weight: float = 0.1,
        act_q_weight: float = 0.5,
        mask_denoise_weight: float = 0.3,
        section_shuffle_weight: float = 0.2,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.doc_loss_weight = doc_loss_weight
        self.span_loss_weight = span_loss_weight
        self.act_ponder_weight = act_ponder_weight
        self.act_q_weight = act_q_weight
        self.mask_denoise_weight = mask_denoise_weight
        self.section_shuffle_weight = section_shuffle_weight
        self.label_smoothing = label_smoothing
        
        # ACT loss computer
        self.act_loss = ACTLoss(ponder_weight=act_ponder_weight)

    def compute_document_loss(
        self, 
        doc_logits: torch.Tensor, 
        doc_labels: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute document classification loss using BCE.
        
        Args:
            doc_logits: [batch, num_doc_scores] - Raw logits
            doc_labels: [batch, num_doc_scores] - Binary labels
            valid_mask: [batch] - Which examples are valid
            
        Returns:
            loss: Scalar BCE loss
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            doc_labels = doc_labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss per example
        bce_loss = F.binary_cross_entropy_with_logits(
            doc_logits, doc_labels.float(), reduction='none'
        )  # [batch, num_doc_scores]
        
        # Average across document scores, sum across valid examples
        loss_per_example = bce_loss.mean(dim=-1)  # [batch]
        total_loss = torch.where(valid_mask, loss_per_example, 0).sum()
        
        return total_loss

    def compute_span_loss(
        self,
        span_logits: torch.Tensor,
        span_labels: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute span tagging loss using cross-entropy.
        
        Args:
            span_logits: [batch, seq_len, num_span_tags] - Raw logits
            span_labels: [batch, seq_len] - Integer labels
            attention_mask: [batch, seq_len] - Which tokens are valid
            
        Returns:
            loss: Scalar cross-entropy loss
        """
        batch_size, seq_len, num_tags = span_logits.shape
        
        # Flatten for cross-entropy
        flat_logits = span_logits.view(-1, num_tags)  # [batch*seq_len, num_tags]
        flat_labels = span_labels.view(-1)  # [batch*seq_len]
        flat_mask = attention_mask.view(-1)  # [batch*seq_len]
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(flat_logits, flat_labels.long(), reduction='none')  # [batch*seq_len]
        
        # Apply mask and sum
        total_loss = torch.where(flat_mask, ce_loss, 0).sum()
        
        return total_loss

    def compute_auxiliary_losses(
        self,
        model_output: HRMDLPOutput,
        batch: Dict[str, torch.Tensor],
        mask_prob: float = 0.15
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for robustness.
        
        Args:
            model_output: Model outputs
            batch: Input batch
            mask_prob: Probability of masking tokens
            
        Returns:
            Dictionary of auxiliary losses
        """
        aux_losses = {}
        
        # For now, return empty dict - auxiliary losses can be added later
        # These would include:
        # 1. Mask-denoise: Predict masked tokens
        # 2. Section-shuffle: Detect shuffled sections
        
        return aux_losses

    def forward(
        self,
        model_output: HRMDLPOutput,
        batch: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total multi-task loss.
        
        Args:
            model_output: Outputs from HRM-DLP model
            batch: Input batch containing labels
            valid_mask: [batch] - Which examples are valid (optional)
            
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components
        """
        device = model_output.doc_logits.device
        batch_size = model_output.doc_logits.shape[0]
        
        # Default valid mask if not provided
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        
        loss_dict = {}
        
        # 1. Document classification loss
        if 'doc_labels' in batch:
            doc_loss = self.compute_document_loss(
                model_output.doc_logits,
                batch['doc_labels'],
                valid_mask
            )
            loss_dict['doc_loss'] = doc_loss
        
        # 2. Span tagging loss  
        if 'bio_labels' in batch:
            span_loss = self.compute_span_loss(
                model_output.span_logits,
                batch['bio_labels'],
                batch.get('attention_mask', torch.ones_like(batch['bio_labels'], dtype=torch.bool))
            )
            loss_dict['span_loss'] = span_loss
        
        # 3. ACT losses
        if hasattr(model_output, 'q_halt_logits'):
            # Determine correctness for ACT (use document accuracy as proxy)
            with torch.no_grad():
                if 'doc_labels' in batch:
                    doc_preds = torch.sigmoid(model_output.doc_logits) > 0.5
                    is_correct = (doc_preds == batch['doc_labels'].bool()).all(dim=-1)
                else:
                    # Fallback: assume all correct if no labels
                    is_correct = torch.ones(batch_size, device=device, dtype=torch.bool)
            
            act_losses = self.act_loss.compute_act_losses(
                model_output.q_halt_logits,
                model_output.q_continue_logits,
                model_output.target_q_continue,
                model_output.steps,
                is_correct,
                valid_mask
            )
            loss_dict.update(act_losses)
        
        # 4. Auxiliary losses
        aux_losses = self.compute_auxiliary_losses(model_output, batch)
        loss_dict.update(aux_losses)
        
        # Combine all losses with weights
        total_loss = 0.0
        
        if 'doc_loss' in loss_dict:
            total_loss += self.doc_loss_weight * loss_dict['doc_loss']
        
        if 'span_loss' in loss_dict:
            total_loss += self.span_loss_weight * loss_dict['span_loss']
        
        if 'ponder_cost' in loss_dict:
            total_loss += loss_dict['ponder_cost']  # Already weighted
        
        if 'q_halt_loss' in loss_dict:
            total_loss += self.act_q_weight * loss_dict['q_halt_loss']
        
        if 'q_continue_loss' in loss_dict:
            total_loss += self.act_q_weight * loss_dict['q_continue_loss']
        
        # Add auxiliary losses with weights
        for key, loss in aux_losses.items():
            if 'mask_denoise' in key:
                total_loss += self.mask_denoise_weight * loss
            elif 'section_shuffle' in key:
                total_loss += self.section_shuffle_weight * loss
        
        return total_loss, loss_dict

    def compute_metrics(
        self,
        model_output: HRMDLPOutput,
        batch: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics.
        
        Args:
            model_output: Model outputs
            batch: Input batch with labels
            valid_mask: Which examples are valid
            
        Returns:
            Dictionary of metrics
        """
        device = model_output.doc_logits.device
        batch_size = model_output.doc_logits.shape[0]
        
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        
        metrics = {}
        
        with torch.no_grad():
            # Document classification metrics
            if 'doc_labels' in batch:
                doc_probs = torch.sigmoid(model_output.doc_logits)
                doc_preds = doc_probs > 0.5
                doc_targets = batch['doc_labels'].bool()
                
                # Per-class accuracy
                correct_per_class = (doc_preds == doc_targets).float()  # [batch, num_classes]
                class_accuracy = torch.where(
                    valid_mask.unsqueeze(-1), correct_per_class, 0
                ).sum(dim=0) / valid_mask.sum()
                
                # Overall document accuracy (all classes correct)
                doc_exact_match = (doc_preds == doc_targets).all(dim=-1)
                metrics['doc_accuracy'] = torch.where(valid_mask, doc_exact_match.float(), 0).sum() / valid_mask.sum()
                
                # Per-class metrics
                for i, class_name in enumerate(['sensitivity', 'exposure', 'context', 'obfuscation']):
                    metrics[f'doc_{class_name}_accuracy'] = class_accuracy[i]
            
            # Span tagging metrics  
            if 'bio_labels' in batch:
                span_preds = model_output.span_logits.argmax(dim=-1)  # [batch, seq_len]
                span_targets = batch['bio_labels']
                attention_mask = batch.get('attention_mask', torch.ones_like(span_targets, dtype=torch.bool))
                
                # Token-level accuracy
                correct_tokens = (span_preds == span_targets) & attention_mask
                total_tokens = attention_mask.sum()
                if total_tokens > 0:
                    metrics['span_token_accuracy'] = correct_tokens.sum().float() / total_tokens
                
                # Sequence-level span accuracy (all spans correct)
                span_exact_match = ((span_preds == span_targets) | ~attention_mask).all(dim=-1)
                metrics['span_sequence_accuracy'] = torch.where(valid_mask, span_exact_match.float(), 0).sum() / valid_mask.sum()
            
            # ACT metrics
            if hasattr(model_output, 'q_halt_logits'):
                # Determine correctness for ACT metrics
                if 'doc_labels' in batch:
                    doc_preds = torch.sigmoid(model_output.doc_logits) > 0.5
                    is_correct = (doc_preds == batch['doc_labels'].bool()).all(dim=-1)
                else:
                    is_correct = torch.ones(batch_size, device=device, dtype=torch.bool)
                
                act_metrics = self.act_loss.compute_act_metrics(
                    model_output.q_halt_logits,
                    model_output.steps,
                    is_correct,
                    valid_mask
                )
                metrics.update(act_metrics)
        
        return metrics


class HRMDLPLossWrapper(nn.Module):
    """
    Wrapper that combines HRM-DLP model with loss computation.
    
    This follows the pattern from the original HRM codebase.
    """
    
    def __init__(self, model: nn.Module, **loss_kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = HRMDLPMultiTaskLoss(**loss_kwargs)
    
    def initial_carry(self, *args, **kwargs):
        """Initialize carry state."""
        return self.model.initial_carry(*args, **kwargs)
    
    def forward(
        self,
        carry,
        batch: Dict[str, torch.Tensor],
        return_keys=None
    ) -> Tuple:
        """
        Forward pass with loss computation.
        
        Returns:
            new_carry: Updated carry state
            total_loss: Combined loss
            metrics: Dictionary of metrics  
            outputs: Model outputs (if requested)
            all_halted: Whether all sequences halted
        """
        if return_keys is None:
            return_keys = []
        
        # Model forward pass
        new_carry, model_output, all_halted = self.model(carry, batch)
        
        # Compute loss and metrics
        valid_mask = new_carry.halted  # Use halted sequences as valid
        total_loss, loss_dict = self.loss_fn(model_output, batch, valid_mask)
        metrics = self.loss_fn.compute_metrics(model_output, batch, valid_mask)
        
        # Add loss components to metrics
        metrics.update({f"loss/{k}": v.detach() for k, v in loss_dict.items()})
        metrics["loss/total"] = total_loss.detach()
        
        # Prepare outputs to return
        outputs = {}
        for key in return_keys:
            if hasattr(model_output, key):
                outputs[key] = getattr(model_output, key).detach()
        
        return new_carry, total_loss, metrics, outputs, all_halted