"""
DLP Tokenizer

Tokenization utilities for DLP training.
"""

from typing import Optional
from pathlib import Path
import sys

# Legacy import compatibility
hrm_path = Path(__file__).parent.parent.parent / "HRM"
sys.path.insert(0, str(hrm_path))

from hrm_dlp.tokenizer import SimpleTokenizer, SentencePieceTokenizer


def create_tokenizer(vocab_size: int, model_path: Optional[str] = None):
    """Create a tokenizer for DLP training"""
    
    if model_path and Path(model_path).exists():
        # Use SentencePiece tokenizer if model exists
        return SentencePieceTokenizer(model_path)
    else:
        # Use simple tokenizer as fallback
        return SimpleTokenizer(vocab_size=vocab_size)