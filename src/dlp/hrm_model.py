"""
HRM-DLP Model: Hierarchical Reasoning Model adapted for Data Loss Prevention tasks.

This model combines:
- Fast/Slow hierarchical reasoning modules from HRM
- DLP-specific output heads (document classification, span tagging, memory)
- ACT (Adaptive Compute Time) for variable computation
- Fusion gates for combining reasoning levels
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn

from .hrm_layers import (
    HRMBlock, HRMReasoningModule, FusionGates, RotaryEmbedding,
    CastedEmbedding, CastedLinear, trunc_normal_init_
)
from .act import ACTController, ACTCarry, ACTLoss
from .dsl import BIO_TAG_TO_ID, ID_TO_BIO_TAG, NUM_BIO_TAGS


@dataclass 
class HRMDLPConfig:
    """Configuration for HRM-DLP model."""
    
    # Model dimensions
    vocab_size: int
    seq_len: int
    hidden_size: int
    
    # HRM architecture
    H_layers: int = 4          # High-level (slow) reasoning layers
    L_layers: int = 4          # Low-level (fast) reasoning layers  
    H_cycles: int = 2          # High-level reasoning cycles
    L_cycles: int = 2          # Low-level reasoning cycles
    
    # Transformer config
    num_heads: int = 6
    expansion: float = 4.0     # MLP expansion factor
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # DLP task heads
    num_doc_scores: int = 4    # sensitivity, exposure, context, obfuscation
    num_span_tags: int = NUM_BIO_TAGS  # BIO tags for span tagging
    memory_vec_dim: int = 256
    
    # ACT configuration
    use_act: bool = True
    act_max_steps: int = 4
    act_exploration_prob: float = 0.0
    
    # Segment processing (like original HRM)
    segment_size: int = 64     # Update slow state every N tokens
    
    # Training
    forward_dtype: str = "bfloat16"


@dataclass
class HRMDLPOutput:
    """Output from HRM-DLP model."""
    
    # DLP task outputs  
    doc_logits: torch.Tensor      # [batch, num_doc_scores] - Document classification
    span_logits: torch.Tensor     # [batch, seq_len, num_span_tags] - Span tagging
    memory_vector: torch.Tensor   # [batch, memory_vec_dim] - Conversation memory
    
    # ACT outputs
    steps: torch.Tensor           # [batch] - Steps taken per sequence
    q_halt_logits: torch.Tensor   # [batch] - Q-values for halting
    q_continue_logits: torch.Tensor  # [batch] - Q-values for continuing
    target_q_continue: Optional[torch.Tensor] = None  # For Q-learning


class HRMDLPInner(nn.Module):
    """
    Inner HRM-DLP model (single reasoning step).
    
    Based on HierarchicalReasoningModel_ACTV1_Inner but adapted for DLP tasks.
    """
    
    def __init__(self, config: HRMDLPConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Token embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, 
            config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=self.forward_dtype
        )
        
        # Position embeddings (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len,
            base=config.rope_theta
        )
        
        # HRM reasoning modules
        self.H_level = HRMReasoningModule(nn.ModuleList([
            HRMBlock(config.hidden_size, config.num_heads, config.expansion, config.rms_norm_eps)
            for _ in range(config.H_layers)
        ]))
        
        self.L_level = HRMReasoningModule(nn.ModuleList([
            HRMBlock(config.hidden_size, config.num_heads, config.expansion, config.rms_norm_eps) 
            for _ in range(config.L_layers)
        ]))
        
        # Fusion gates for combining input/fast/slow
        self.fusion_gates = FusionGates(config.hidden_size)
        
        # Initial states for H and L levels
        self.H_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1)
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1) 
        )
        
        # DLP task heads
        self.doc_head = CastedLinear(config.hidden_size, config.num_doc_scores, bias=True)
        self.span_head = CastedLinear(config.hidden_size, config.num_span_tags, bias=True) 
        self.memory_head = CastedLinear(config.hidden_size, config.memory_vec_dim, bias=True)
        
        # ACT controller
        if config.use_act:
            self.act_controller = ACTController(
                config.hidden_size, 
                config.act_max_steps,
                config.act_exploration_prob
            )

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute input embeddings with scaling."""
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    def _segment_pooling(self, states: torch.Tensor) -> torch.Tensor:
        """
        Pool fast states into segments for slow processing.
        
        Args:
            states: [batch, seq_len, hidden_size] - Fast reasoning states
            
        Returns:
            pooled: [batch, num_segments, hidden_size] - Pooled segment representations
        """
        batch_size, seq_len, hidden_size = states.shape
        segment_size = self.config.segment_size
        
        # Pad sequence to multiple of segment_size
        if seq_len % segment_size != 0:
            pad_size = segment_size - (seq_len % segment_size)
            states = F.pad(states, (0, 0, 0, pad_size))
            seq_len = states.shape[1]
        
        num_segments = seq_len // segment_size
        
        # Reshape and pool
        states_reshaped = states.view(batch_size, num_segments, segment_size, hidden_size)
        
        # Use mean + max pooling for richer representation
        mean_pooled = states_reshaped.mean(dim=2)  # [batch, num_segments, hidden_size]
        max_pooled = states_reshaped.max(dim=2)[0]
        
        # Combine mean and max (simple concatenation then projection)
        combined = mean_pooled + max_pooled  # Simple addition for now
        
        return combined

    def _expand_segments_to_tokens(self, segment_states: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        """
        Expand segment-level states back to token-level for fusion.
        
        Args:
            segment_states: [batch, num_segments, hidden_size]
            target_seq_len: Target sequence length
            
        Returns:
            expanded: [batch, seq_len, hidden_size]
        """
        batch_size, num_segments, hidden_size = segment_states.shape
        segment_size = self.config.segment_size
        
        # Repeat each segment for all tokens in that segment
        expanded = segment_states.unsqueeze(2).expand(batch_size, num_segments, segment_size, hidden_size)
        expanded = expanded.contiguous().view(batch_size, num_segments * segment_size, hidden_size)
        
        # Truncate to target length if necessary
        if expanded.shape[1] > target_seq_len:
            expanded = expanded[:, :target_seq_len]
        elif expanded.shape[1] < target_seq_len:
            # Pad if necessary (shouldn't happen in normal operation)
            pad_size = target_seq_len - expanded.shape[1]
            expanded = F.pad(expanded, (0, 0, 0, pad_size))
        
        return expanded

    def forward(
        self, 
        carry: ACTCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[ACTCarry, HRMDLPOutput]:
        """
        Forward pass of HRM-DLP inner model.
        
        Args:
            carry: ACT carry state
            batch: Input batch with 'input_ids' and other data
            
        Returns:
            new_carry: Updated carry state
            output: Model outputs
        """
        input_ids = batch['input_ids']  # [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Get RoPE embeddings
        cos, sin = self.rotary_emb()
        seq_info = {'cos_sin': (cos, sin)}
        
        # Input embeddings
        input_embeddings = self._input_embeddings(input_ids)  # [batch, seq_len, hidden_size]
        
        # Initialize states if needed
        if carry.fast_state.numel() == 0:
            # First step - initialize with learned initial states
            z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        else:
            z_H = carry.fast_state   # Fast (high-frequency) state
            z_L = carry.slow_state   # Slow (low-frequency) state
        
        # HRM reasoning cycles (no-grad for stability except final step)  
        with torch.no_grad():
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    # Skip final step to allow gradients
                    if not ((h_step == self.config.H_cycles - 1) and (l_step == self.config.L_cycles - 1)):
                        # Fast reasoning: process all tokens
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                
                # Slow reasoning: work at segment level
                if not (h_step == self.config.H_cycles - 1):
                    # Pool fast states to segments  
                    pooled_L = self._segment_pooling(z_L)  # [batch, num_segments, hidden_size]
                    
                    # Apply slow reasoning to segments
                    slow_input = self._segment_pooling(z_H + input_embeddings)
                    pooled_H = self._segment_pooling(z_H)
                    
                    updated_pooled_H = self.H_level(pooled_H, slow_input, **seq_info)
                    
                    # Expand back to token level
                    z_H = self._expand_segments_to_tokens(updated_pooled_H, seq_len)

        # Final step with gradients (1-step gradient principle)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        
        # Apply fusion gates
        fused_repr = self.fusion_gates(input_embeddings, z_L, z_H)
        
        # DLP task heads
        # Document-level: use CLS token (first token) for document classification
        doc_repr = fused_repr[:, 0]  # [batch, hidden_size]
        doc_logits = self.doc_head(doc_repr)  # [batch, num_doc_scores]
        
        # Token-level: span tagging for all tokens
        span_logits = self.span_head(fused_repr)  # [batch, seq_len, num_span_tags]
        
        # Memory vector: global document representation
        memory_vector = self.memory_head(doc_repr)  # [batch, memory_vec_dim]
        
        # ACT decisions
        q_halt_logits = torch.zeros(batch_size, device=input_ids.device)
        q_continue_logits = torch.zeros(batch_size, device=input_ids.device)
        target_q_continue = None
        
        if self.config.use_act and hasattr(self, 'act_controller'):
            halt_decisions, q_halt_logits, q_continue_logits = self.act_controller.should_halt(
                fused_repr, carry.steps, self.training
            )
            
            # Compute target Q for next step if in training
            if self.training and self.config.act_max_steps > 1:
                # Run one more step to get target Q
                next_fused_repr = self.fusion_gates(input_embeddings, z_L, z_H)  # Simple approximation
                is_final_step = (carry.steps + 1) >= self.config.act_max_steps
                target_q_continue = self.act_controller.compute_target_q(
                    next_fused_repr, carry.steps, is_final_step
                )
        else:
            halt_decisions = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Update carry for next step
        new_carry = ACTCarry(
            fast_state=z_L.detach(),
            slow_state=z_H.detach(),
            steps=carry.steps + 1,
            halted=halt_decisions,
            current_data=carry.current_data
        )
        
        output = HRMDLPOutput(
            doc_logits=doc_logits,
            span_logits=span_logits,
            memory_vector=memory_vector,
            steps=carry.steps + 1,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            target_q_continue=target_q_continue
        )
        
        return new_carry, output


class HRMDLP(nn.Module):
    """
    Main HRM-DLP model with ACT wrapper.
    
    Provides the full ACT loop over the inner model.
    """
    
    def __init__(self, config: HRMDLPConfig):
        super().__init__()
        self.config = config
        self.inner = HRMDLPInner(config)
        
        if config.use_act:
            self.act_controller = ACTController(
                config.hidden_size,
                config.act_max_steps, 
                config.act_exploration_prob
            )

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ACTCarry:
        """Initialize ACT carry for a new batch."""
        input_ids = batch['input_ids']
        batch_size, seq_len = input_ids.shape
        
        if self.config.use_act and hasattr(self, 'act_controller'):
            return self.act_controller.initial_carry(
                batch_size, seq_len, input_ids.device, self.inner.forward_dtype
            )
        else:
            # Non-ACT mode: single step
            return ACTCarry(
                fast_state=torch.zeros(batch_size, seq_len, self.config.hidden_size, 
                                     device=input_ids.device, dtype=self.inner.forward_dtype),
                slow_state=torch.zeros(batch_size, seq_len, self.config.hidden_size,
                                     device=input_ids.device, dtype=self.inner.forward_dtype),
                steps=torch.zeros(batch_size, device=input_ids.device, dtype=torch.int32),
                halted=torch.ones(batch_size, device=input_ids.device, dtype=torch.bool),
                current_data=batch
            )

    def forward(self, carry: ACTCarry, batch: Dict[str, torch.Tensor]) -> Tuple[ACTCarry, HRMDLPOutput, bool]:
        """
        Forward pass with ACT loop.
        
        Args:
            carry: Current ACT carry state
            batch: Input batch
            
        Returns:
            new_carry: Updated carry state  
            output: Model output from last step
            all_halted: Whether all sequences have halted
        """
        # Update carry for new batch data (reset halted sequences)
        if self.config.use_act and hasattr(self, 'act_controller'):
            carry = self.act_controller.reset_carry_for_halted(carry, batch)
        
        # Single reasoning step
        carry, output = self.inner(carry, batch)
        
        # Check if all sequences have halted
        all_halted = carry.halted.all()
        
        return carry, output, all_halted


def create_hrm_dlp_model(config_dict: Dict) -> HRMDLP:
    """Factory function to create HRM-DLP model from config dictionary."""
    config = HRMDLPConfig(**config_dict)
    return HRMDLP(config)