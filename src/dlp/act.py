"""
Adaptive Compute Time (ACT) implementation for HRM-DLP.
Provides variable computation based on input complexity with Q-learning.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .hrm_layers import CastedLinear, trunc_normal_init_


@dataclass
class ACTCarry:
    """State carried between ACT steps."""
    # HRM internal states
    fast_state: torch.Tensor      # [batch, seq_len, hidden_size]
    slow_state: torch.Tensor      # [batch, seq_len, hidden_size]
    
    # ACT control
    steps: torch.Tensor           # [batch] - number of steps taken
    halted: torch.Tensor         # [batch] - whether each sequence has halted
    
    # Current batch data
    current_data: Dict[str, torch.Tensor]


class ACTController(nn.Module):
    """
    ACT controller using Q-learning for halt decisions.
    
    Based on the original HRM implementation with adaptations for DLP tasks.
    """
    
    def __init__(self, hidden_size: int, max_steps: int = 4, exploration_prob: float = 0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.exploration_prob = exploration_prob
        
        # Q-network for halt decisions
        # Takes the first token (CLS-like) representation and outputs Q-values for [halt, continue]
        self.q_head = CastedLinear(hidden_size, 2, bias=True)
        
        # Initialize Q to favor continuation initially (speeds up learning)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # Start with strong bias toward not halting

    def should_halt(self, representations: torch.Tensor, steps: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decide whether to halt computation for each sequence.
        
        Args:
            representations: [batch, seq_len, hidden_size] - Current hidden states
            steps: [batch] - Current step count for each sequence
            training: Whether we're in training mode
            
        Returns:
            halt_decisions: [batch] - Boolean tensor indicating which sequences should halt
            q_halt_logits: [batch] - Q-values for halting
            q_continue_logits: [batch] - Q-values for continuing
        """
        batch_size = representations.shape[0]
        
        # Use first token as sequence representation (CLS-like)
        cls_repr = representations[:, 0]  # [batch, hidden_size]
        
        # Get Q-values
        q_logits = self.q_head(cls_repr)  # [batch, 2]
        q_halt_logits = q_logits[:, 0]     # [batch]
        q_continue_logits = q_logits[:, 1]  # [batch]
        
        # Halting decisions
        with torch.no_grad():
            # Always halt if max steps reached
            max_steps_reached = steps >= self.max_steps
            
            # During training, use Q-values for halting
            if training and self.max_steps > 1:
                # Basic Q-learning: halt if Q(halt) > Q(continue)
                q_based_halt = q_halt_logits > q_continue_logits
                
                # Add exploration: sometimes force minimum steps
                if self.exploration_prob > 0:
                    explore_mask = torch.rand_like(q_halt_logits) < self.exploration_prob
                    min_steps = torch.randint_like(steps, low=2, high=self.max_steps + 1)
                    exploration_continue = explore_mask & (steps < min_steps)
                    q_based_halt = q_based_halt & ~exploration_continue
                
                halt_decisions = max_steps_reached | q_based_halt
            else:
                # During evaluation, always use max steps for consistent batching
                halt_decisions = max_steps_reached
        
        return halt_decisions, q_halt_logits, q_continue_logits

    def compute_target_q(self, next_representations: torch.Tensor, steps: torch.Tensor, is_final_step: torch.Tensor) -> torch.Tensor:
        """
        Compute target Q-values for Q-learning update.
        
        Uses a simple approach without replay buffers - relies on large batch size
        for stability (similar to PQN - Population Q-Networks).
        
        Args:
            next_representations: [batch, seq_len, hidden_size] - Next step representations
            steps: [batch] - Current step count
            is_final_step: [batch] - Whether this is the final step
            
        Returns:
            target_q_continue: [batch] - Target Q-values for continue action
        """
        with torch.no_grad():
            # Get Q-values for next state
            next_cls_repr = next_representations[:, 0]
            next_q_logits = self.q_head(next_cls_repr)
            next_q_halt = next_q_logits[:, 0]
            next_q_continue = next_q_logits[:, 1]
            
            # For final steps, target is halt Q-value
            # For non-final steps, target is max(halt, continue) 
            target_q_continue = torch.where(
                is_final_step, 
                next_q_halt,
                torch.maximum(next_q_halt, next_q_continue)
            )
            
            # Apply sigmoid to get values in [0,1] range
            target_q_continue = torch.sigmoid(target_q_continue)
            
        return target_q_continue

    def initial_carry(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> ACTCarry:
        """Initialize ACT carry state for a new batch."""
        return ACTCarry(
            fast_state=torch.zeros(batch_size, seq_len, self.hidden_size, device=device, dtype=dtype),
            slow_state=torch.zeros(batch_size, seq_len, self.hidden_size, device=device, dtype=dtype),
            steps=torch.zeros(batch_size, device=device, dtype=torch.int32),
            halted=torch.ones(batch_size, device=device, dtype=torch.bool),  # Start halted
            current_data={}
        )

    def reset_carry_for_halted(self, carry: ACTCarry, batch_data: Dict[str, torch.Tensor]) -> ACTCarry:
        """Reset carry state for sequences that have halted."""
        # Reset steps for halted sequences
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        
        # Update current data for halted sequences  
        new_current_data = {}
        for key, value in batch_data.items():
            # For halted sequences, use new batch data; for continuing, keep current
            expand_dims = (1,) * (value.ndim - 1)  # Handle multi-dimensional tensors
            halted_expanded = carry.halted.view(-1, *expand_dims)
            new_current_data[key] = torch.where(halted_expanded, value, carry.current_data.get(key, value))
        
        return ACTCarry(
            fast_state=carry.fast_state,  # These will be reset by the model
            slow_state=carry.slow_state,
            steps=new_steps,
            halted=torch.zeros_like(carry.halted),  # All sequences are now active
            current_data=new_current_data
        )


class ACTLoss(nn.Module):
    """
    Loss functions for Adaptive Compute Time.
    
    Combines task loss with ACT-specific losses:
    - Ponder cost: Penalizes excessive computation
    - Q-learning loss: Trains the halt decision network
    """
    
    def __init__(self, ponder_weight: float = 0.1):
        super().__init__()
        self.ponder_weight = ponder_weight

    def compute_act_losses(
        self, 
        q_halt_logits: torch.Tensor,
        q_continue_logits: torch.Tensor, 
        target_q_continue: torch.Tensor,
        steps: torch.Tensor,
        is_correct: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ACT-specific losses.
        
        Args:
            q_halt_logits: [batch] - Q-values for halting
            q_continue_logits: [batch] - Q-values for continuing  
            target_q_continue: [batch] - Target Q-values for continue action
            steps: [batch] - Number of steps taken
            is_correct: [batch] - Whether the prediction is correct
            valid_mask: [batch] - Which examples are valid (not padding)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Ponder cost - penalize taking many steps
        ponder_cost = self.ponder_weight * steps.float()
        losses['ponder_cost'] = torch.where(valid_mask, ponder_cost, 0).sum()
        
        # Q-learning loss for halt decisions
        # We want to halt when prediction is correct, continue when incorrect
        halt_targets = is_correct.float()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, halt_targets, reduction='none'
        )
        losses['q_halt_loss'] = torch.where(valid_mask, q_halt_loss, 0).sum()
        
        # Q-learning loss for continue decisions (bootstrapping)
        if target_q_continue is not None:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits, target_q_continue, reduction='none'  
            )
            losses['q_continue_loss'] = torch.where(valid_mask, q_continue_loss, 0).sum()
        
        return losses

    def compute_act_metrics(
        self,
        q_halt_logits: torch.Tensor,
        steps: torch.Tensor, 
        is_correct: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute ACT-specific metrics for logging."""
        with torch.no_grad():
            metrics = {}
            
            if valid_mask.sum() > 0:
                # Average steps taken
                metrics['avg_steps'] = torch.where(valid_mask, steps.float(), 0).sum() / valid_mask.sum()
                
                # Q-network accuracy (does it correctly predict when to halt?)
                halt_predictions = (q_halt_logits >= 0)  # Positive logit = predict halt
                halt_targets = is_correct
                q_accuracy = (halt_predictions == halt_targets).float()
                metrics['q_halt_accuracy'] = torch.where(valid_mask, q_accuracy, 0).sum() / valid_mask.sum()
                
            return metrics