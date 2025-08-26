#!/usr/bin/env python3
"""Train a SentencePiece tokenizer for HRM-DLP on the existing data"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dlp.tokenizer import create_tokenizer, TokenizerConfig


def main():
    # Data paths from config
    data_dir = "data/runs/run_20250824_123640_a2f52bf9/split_outputs"
    train_file = f"{data_dir}/train_examples_augmented.jsonl"
    val_file = f"{data_dir}/val_examples_augmented.jsonl"
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: Training file not found: {train_file}")
        return
    if not os.path.exists(val_file):
        print(f"Error: Validation file not found: {val_file}")
        return
    
    print(f"Training tokenizer on:")
    print(f"  - Training data: {train_file}")
    print(f"  - Validation data: {val_file}")
    
    # Create output directory
    tokenizer_dir = "tokenizers"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Tokenizer configuration
    config = TokenizerConfig(
        vocab_size=16000,
        model_type="bpe",
        character_coverage=0.9995
    )
    
    # Train tokenizer
    model_prefix = f"{tokenizer_dir}/dlp_tokenizer"
    
    print(f"\nTraining SentencePiece tokenizer...")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Model type: {config.model_type}")
    print(f"  - Output prefix: {model_prefix}")
    
    tokenizer = create_tokenizer(
        jsonl_files=[train_file, val_file],
        model_prefix=model_prefix,
        config=config
    )
    
    print(f"\n✅ Tokenizer training completed!")
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test tokenizer
    test_text = "<CHANNEL email><USER role=LEGAL dept=CORP><SUBJECT>Test Email</SUBJECT><BODY>This is a test message with PII like john.doe@company.com</BODY>"
    tokens = tokenizer.encode(test_text)
    pieces = tokenizer.encode_pieces(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\n🧪 Tokenizer test:")
    print(f"Input: {test_text}")
    print(f"Tokens ({len(tokens)}): {tokens[:20]}..." if len(tokens) > 20 else f"Tokens: {tokens}")
    print(f"Pieces: {pieces[:10]}..." if len(pieces) > 10 else f"Pieces: {pieces}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {test_text == decoded}")


if __name__ == "__main__":
    main()