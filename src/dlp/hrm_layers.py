"""
HRM-specific layers adapted from the original HRM implementation.
Contains Fast/Slow modules, attention, and fusion gates for DLP tasks.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    """Helper function to find the smallest multiple of b that is >= a."""
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to queries and keys."""
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


class CastedLinear(nn.Module):
    """Linear layer with dynamic casting and truncated normal init."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    """Embedding layer with dynamic casting and custom initialization."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention with optional RoPE and fallback when flash attention unavailable."""
    
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Fallback attention (since flash attention may not be available)
        # Reshape for standard attention
        query = query.transpose(1, 2)  # [bs, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.causal:
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
            attn_weights.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [bs, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU activation function - Swish-Gated Linear Unit."""
    
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Root Mean Square Layer Normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class HRMBlock(nn.Module):
    """Single HRM transformer block with post-norm residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, rms_norm_eps: float = 1e-5):
        super().__init__()

        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm - Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Post Norm - MLP
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HRMReasoningModule(nn.Module):
    """HRM reasoning module containing multiple transformer blocks with input injection."""
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class FusionGates(nn.Module):
    """Learned fusion gates for combining input, fast, and slow representations."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        # Gate network: takes [input || fast || slow] and outputs gating weights
        self.gate_proj = CastedLinear(hidden_size * 3, hidden_size, bias=True)
        self.fast_transform = CastedLinear(hidden_size, hidden_size, bias=False)
        self.slow_transform = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, input_emb: torch.Tensor, fast_state: torch.Tensor, slow_state: torch.Tensor) -> torch.Tensor:
        """
        Fuse input, fast, and slow representations using learned gates.
        
        Args:
            input_emb: [batch, seq_len, hidden_size] - Input token embeddings
            fast_state: [batch, seq_len, hidden_size] - Fast reasoning state
            slow_state: [batch, seq_len, hidden_size] - Slow reasoning state (broadcasted from segments)
            
        Returns:
            fused: [batch, seq_len, hidden_size] - Fused representation
        """
        # Concatenate for gating
        combined = torch.cat([input_emb, fast_state, slow_state], dim=-1)
        
        # Compute gate (sigmoid to get values in [0,1])
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # Apply transformations and gating
        fast_transformed = self.fast_transform(fast_state)
        slow_transformed = self.slow_transform(slow_state)
        
        # Gated combination: gate controls fast vs slow, input is always included
        fused = input_emb + gate * fast_transformed + (1 - gate) * slow_transformed
        
        return fused