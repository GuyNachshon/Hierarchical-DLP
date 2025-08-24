"""Loss functions for HRM-DLP multi-task training

Implements the combined loss function:
L = BCE(doc) + CE(BIO) + 0.3*MaskDenoise + 0.2*SectionShuffle
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import random
import math


class DLPLossConfig:
    """Configuration for DLP loss computation"""
    def __init__(
        self,
        doc_loss_weight: float = 1.0,
        span_loss_weight: float = 1.0,
        mask_denoise_weight: float = 0.3,
        section_shuffle_weight: float = 0.2,
        label_smoothing: float = 0.05,
        ignore_index: int = -100,
        mask_prob: float = 0.15,
        shuffle_prob: float = 0.10
    ):
        self.doc_loss_weight = doc_loss_weight
        self.span_loss_weight = span_loss_weight
        self.mask_denoise_weight = mask_denoise_weight
        self.section_shuffle_weight = section_shuffle_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob


class DocumentClassificationLoss(nn.Module):
    """Binary Cross-Entropy loss for document-level classification"""
    
    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] - raw logits
            labels: [batch_size, num_labels] - binary labels (0 or 1)
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        return loss


class SpanTaggingLoss(nn.Module):
    """Cross-entropy loss for BIO span tagging"""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, seq_len, num_tags] - span tag logits
            labels: [batch_size, seq_len] - span tag labels
        """
        # Flatten for cross-entropy computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.ignore_index, reduction='mean')
        return loss


class MaskDenoiseLoss(nn.Module):
    """Auxiliary loss for masked language modeling (denoising)"""
    
    def __init__(self, vocab_size: int, mask_prob: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = 1  # Assume mask token ID
        
    def create_masked_inputs(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked inputs for denoising task
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            masked_input_ids: Input with some tokens masked
            mask_positions: Boolean mask indicating masked positions
            original_tokens: Original tokens at masked positions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create random mask
        mask_positions = torch.rand(batch_size, seq_len, device=device) < self.mask_prob
        
        # Don't mask special tokens (assume first few tokens are special)
        mask_positions[:, :5] = False  # Don't mask first 5 tokens (likely special tokens)
        
        # Create masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = self.mask_token_id
        
        return masked_input_ids, mask_positions, input_ids
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, vocab_projection: nn.Module) -> torch.Tensor:
        """
        Compute mask-denoising loss
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len] - original input tokens
            vocab_projection: Linear layer for vocab prediction
        """
        # Create masked version
        masked_inputs, mask_positions, original_tokens = self.create_masked_inputs(input_ids)
        
        if not mask_positions.any():
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        
        # Get logits for masked positions
        vocab_logits = vocab_projection(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # Extract logits and labels for masked positions only
        masked_logits = vocab_logits[mask_positions]  # [num_masked, vocab_size]
        masked_labels = original_tokens[mask_positions]  # [num_masked]
        
        # Cross-entropy loss
        loss = F.cross_entropy(masked_logits, masked_labels, reduction='mean')
        return loss


class SectionShuffleLoss(nn.Module):
    """Auxiliary loss for detecting shuffled sections"""
    
    def __init__(self, shuffle_prob: float = 0.10, hidden_size: int = 384):
        super().__init__()
        self.shuffle_prob = shuffle_prob
        self.classifier = nn.Linear(hidden_size, 2)  # Binary: shuffled or not
        
    def create_shuffled_inputs(self, input_ids: torch.Tensor, segment_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create shuffled versions of input sequences
        
        Args:
            input_ids: [batch_size, seq_len]
            segment_size: Size of segments to shuffle
            
        Returns:
            shuffled_inputs: Input with some segments shuffled
            shuffle_labels: Binary labels (1 if shuffled, 0 if not)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        shuffled_inputs = input_ids.clone()
        shuffle_labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        for i in range(batch_size):
            if random.random() < self.shuffle_prob:
                # Shuffle segments
                num_segments = seq_len // segment_size
                if num_segments > 1:
                    # Create segments
                    segments = []
                    for j in range(num_segments):
                        start_idx = j * segment_size
                        end_idx = min((j + 1) * segment_size, seq_len)
                        segments.append(input_ids[i, start_idx:end_idx])
                    
                    # Shuffle segments
                    random.shuffle(segments)
                    
                    # Reconstruct sequence
                    shuffled_seq = torch.cat(segments, dim=0)
                    if shuffled_seq.size(0) < seq_len:
                        # Pad if necessary
                        padding = input_ids[i, shuffled_seq.size(0):]
                        shuffled_seq = torch.cat([shuffled_seq, padding], dim=0)
                    
                    shuffled_inputs[i] = shuffled_seq[:seq_len]
                    shuffle_labels[i] = 1
        
        return shuffled_inputs, shuffle_labels
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute section shuffle detection loss
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
        """
        # Create shuffled inputs and labels
        shuffled_inputs, shuffle_labels = self.create_shuffled_inputs(input_ids)
        
        # Use pooled representation for classification
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Binary classification
        logits = self.classifier(pooled_hidden)  # [batch_size, 2]
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, shuffle_labels, reduction='mean')
        return loss


class DLPMultiTaskLoss(nn.Module):
    """Combined multi-task loss for DLP training"""
    
    def __init__(self, config: DLPLossConfig, vocab_size: int, hidden_size: int = 384):
        super().__init__()
        self.config = config
        
        # Primary losses
        self.doc_loss = DocumentClassificationLoss(config.label_smoothing)
        self.span_loss = SpanTaggingLoss(config.ignore_index)
        
        # Auxiliary losses
        self.mask_denoise_loss = MaskDenoiseLoss(vocab_size, config.mask_prob)
        self.section_shuffle_loss = SectionShuffleLoss(config.shuffle_prob, hidden_size)
        
        # Vocabulary projection for auxiliary tasks
        self.vocab_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self,
        doc_logits: torch.Tensor,
        span_logits: torch.Tensor, 
        hidden_states: torch.Tensor,
        doc_labels: torch.Tensor,
        span_labels: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-task loss
        
        Args:
            doc_logits: [batch_size, num_doc_labels]
            span_logits: [batch_size, seq_len, num_span_tags]
            hidden_states: [batch_size, seq_len, hidden_size]
            doc_labels: [batch_size, num_doc_labels]
            span_labels: [batch_size, seq_len]
            input_ids: [batch_size, seq_len]
            
        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        
        # Primary losses
        losses['doc_loss'] = self.doc_loss(doc_logits, doc_labels)
        losses['span_loss'] = self.span_loss(span_logits, span_labels)
        
        # Auxiliary losses
        losses['mask_denoise_loss'] = self.mask_denoise_loss(hidden_states, input_ids, self.vocab_proj)
        losses['section_shuffle_loss'] = self.section_shuffle_loss(hidden_states, input_ids)
        
        # Combined loss
        total_loss = (
            self.config.doc_loss_weight * losses['doc_loss'] +
            self.config.span_loss_weight * losses['span_loss'] +
            self.config.mask_denoise_weight * losses['mask_denoise_loss'] +
            self.config.section_shuffle_weight * losses['section_shuffle_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


def create_dlp_loss(vocab_size: int, hidden_size: int = 384, **kwargs) -> DLPMultiTaskLoss:
    """Create DLP multi-task loss with configuration"""
    config = DLPLossConfig(**kwargs)
    loss_fn = DLPMultiTaskLoss(config, vocab_size, hidden_size)
    return loss_fn