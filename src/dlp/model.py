"""HRM-DLP Model: Hierarchical Reasoning Model adapted for Data Loss Prevention

This module adapts the original HRM architecture to include DLP-specific heads:
- Document-level classification (sensitivity, exposure, context, obfuscation)
- Token-level BIO span tagging
- Memory summary generation
"""

from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from .hrm_layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, trunc_normal_init_


class DLPModelConfig(BaseModel):
    """Configuration for HRM-DLP model"""
    batch_size: int
    seq_len: int = 1024
    vocab_size: int = 16000
    
    # Hierarchical structure
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    
    # Transformer architecture
    hidden_size: int = 384
    expansion: float = 4.0
    num_heads: int = 6
    pos_encodings: str = "rope"  # or "learned"
    
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # DLP-specific heads
    num_doc_labels: int = 4  # sensitivity, exposure, context, obfuscation
    num_bio_tags: int = 21   # BIO tags for spans
    memory_dim: int = 256    # Memory summary vector dimension
    
    # Fusion gates
    use_fusion_gates: bool = True
    
    # Deterministic mode (disable ACT for production)
    use_act: bool = False
    halt_max_steps: int = 1
    
    forward_dtype: str = "bfloat16"


@dataclass
class DLPModelOutput:
    """Output from DLP model forward pass"""
    doc_logits: torch.Tensor        # [batch_size, num_doc_labels]
    span_logits: torch.Tensor       # [batch_size, seq_len, num_bio_tags]
    memory_vector: torch.Tensor     # [batch_size, memory_dim]
    hidden_states: torch.Tensor     # [batch_size, seq_len, hidden_size]


class FusionGate(nn.Module):
    """Learned fusion gate for combining fast/slow representations"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size * 3, 1)  # input || fast || slow
        
    def forward(self, input_emb: torch.Tensor, fast_hidden: torch.Tensor, slow_hidden: torch.Tensor) -> torch.Tensor:
        """Apply fusion gate to combine representations"""
        # Concatenate all three representations
        combined = torch.cat([input_emb, fast_hidden, slow_hidden], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # Gated combination: gate * fast + (1-gate) * slow
        return gate * fast_hidden + (1 - gate) * slow_hidden


class DocumentScoreHead(nn.Module):
    """Classification head for document-level scores"""
    
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use [CLS] token (first token) for classification
        cls_hidden = hidden_states[:, 0]  # [batch_size, hidden_size]
        cls_hidden = self.norm(cls_hidden)
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)
        return logits


class SpanTaggingHead(nn.Module):
    """Token-level BIO tagging head"""
    
    def __init__(self, hidden_size: int, num_tags: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply to all tokens
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_tags]
        return logits


class MemorySummaryHead(nn.Module):
    """Head for generating memory summary vectors"""
    
    def __init__(self, hidden_size: int, memory_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projector = nn.Linear(hidden_size, memory_dim)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        pooled = self.norm(pooled)
        memory_vec = self.activation(self.projector(pooled))
        return memory_vec


class DLPReasoningBlock(nn.Module):
    """Single reasoning block for DLP model (based on HRM block)"""
    
    def __init__(self, config: DLPModelConfig):
        super().__init__()
        self.config = config
        
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Non-causal for DLP
        )
        
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        
        self.norm_eps = config.rms_norm_eps
        
    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        # Post-norm architecture
        # Self Attention
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        
        # Feed Forward
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        
        return hidden_states


class DLPReasoningModule(nn.Module):
    """Multi-layer reasoning module"""
    
    def __init__(self, config: DLPModelConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DLPReasoningBlock(config) for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        # Input injection (residual connection)
        hidden_states = hidden_states + input_injection
        
        # Apply all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
            
        return hidden_states


class HRMDLPModel(nn.Module):
    """Main HRM-DLP model combining hierarchical reasoning with DLP heads"""
    
    def __init__(self, config: DLPModelConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        # Embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, 
            config.hidden_size, 
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )
        
        # Position embeddings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len,
                base=config.rope_theta
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError(f"Position encoding {config.pos_encodings} not implemented")
        
        # Hierarchical reasoning modules
        self.H_level = DLPReasoningModule(config, config.H_layers)  # High-level (slow)
        self.L_level = DLPReasoningModule(config, config.L_layers)  # Low-level (fast)
        
        # Initial states for hierarchical levels
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1))
        
        # Fusion gates (optional)
        if config.use_fusion_gates:
            self.fusion_gate = FusionGate(config.hidden_size)
        
        # DLP-specific heads
        self.doc_head = DocumentScoreHead(config.hidden_size, config.num_doc_labels)
        self.span_head = SpanTaggingHead(config.hidden_size, config.num_bio_tags)
        self.memory_head = MemorySummaryHead(config.hidden_size, config.memory_dim)
        
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token and position embeddings"""
        # Token embeddings
        embeddings = self.embed_tokens(input_ids.to(torch.int32))
        
        # Position embeddings
        if self.config.pos_encodings == "learned":
            pos_embeddings = self.embed_pos.weight[:input_ids.size(1)]
            # Scale by 1/sqrt(2) to maintain variance
            embeddings = 0.707106781 * (embeddings + pos_embeddings)
        
        # Scale embeddings
        return self.embed_scale * embeddings
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> DLPModelOutput:
        """
        Forward pass through HRM-DLP model
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            
        Returns:
            DLPModelOutput with doc_logits, span_logits, memory_vector
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        input_embeddings = self._get_embeddings(input_ids)
        
        # Initialize hidden states
        device = input_ids.device
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Get positional information
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb()
        else:
            cos_sin = None
        
        # Hierarchical reasoning cycles
        for h_step in range(self.config.H_cycles):
            for l_step in range(self.config.L_cycles):
                # Low-level (fast) reasoning
                if not ((h_step == self.config.H_cycles - 1) and (l_step == self.config.L_cycles - 1)):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            
            # High-level (slow) reasoning  
            if not (h_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        
        # Final update
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        
        # Optional fusion
        if hasattr(self, 'fusion_gate'):
            hidden_states = self.fusion_gate(input_embeddings, z_L, z_H)
        else:
            hidden_states = z_H  # Use high-level states by default
        
        # Apply attention mask if provided
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        
        # DLP heads
        doc_logits = self.doc_head(hidden_states)
        span_logits = self.span_head(hidden_states)
        memory_vector = self.memory_head(hidden_states)
        
        return DLPModelOutput(
            doc_logits=doc_logits,
            span_logits=span_logits,
            memory_vector=memory_vector,
            hidden_states=hidden_states
        )


def create_dlp_model(config_dict: Dict[str, Any]) -> HRMDLPModel:
    """Create DLP model from configuration dictionary"""
    config = DLPModelConfig(**config_dict)
    model = HRMDLPModel(config)
    return model