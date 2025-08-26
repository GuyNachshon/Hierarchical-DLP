#!/usr/bin/env python3
"""Debug script to analyze training data and identify issues"""

import sys
import os
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import torch first to avoid flash attention issues
import torch

# Mock flash attention modules to avoid import errors
class MockModule:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            raise NotImplementedError("Flash attention not available")
        return mock_func

# Mock the problematic imports
sys.modules['flash_attn_interface'] = MockModule()
sys.modules['flash_attn'] = MockModule()

from src.dlp.dataset import DLPDataset, DLPDatasetConfig
from src.dlp.tokenizer import DLPTokenizer


def analyze_jsonl_file(file_path):
    """Analyze raw JSONL data to understand label distributions"""
    print(f"\n🔍 Analyzing {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    doc_labels = defaultdict(list)
    span_types = Counter()
    attachment_formats = {"string": 0, "dict": 0, "list": 0}
    link_formats = {"string": 0, "dict": 0, "list": 0, "none": 0}
    
    total_examples = 0
    valid_examples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            total_examples += 1
            
            try:
                data = json.loads(line.strip())
                valid_examples += 1
                
                # Analyze document labels
                if "labels" in data:
                    labels = data["labels"]
                    if isinstance(labels, dict):
                        for key, value in labels.items():
                            doc_labels[key].append(value)
                    else:
                        print(f"⚠️  Line {line_num}: labels is not a dict: {type(labels)}")
                
                # Analyze spans
                if "spans" in data:
                    spans = data["spans"]
                    if isinstance(spans, list):
                        for span in spans:
                            if isinstance(span, dict) and "type" in span:
                                span_types[span["type"]] += 1
                
                # Analyze attachment formats
                if "attachments" in data:
                    att = data["attachments"]
                    if isinstance(att, str):
                        attachment_formats["string"] += 1
                    elif isinstance(att, list):
                        attachment_formats["list"] += 1
                    elif isinstance(att, dict):
                        attachment_formats["dict"] += 1
                
                # Analyze link formats  
                if "links" in data:
                    links = data["links"]
                    if links is None:
                        link_formats["none"] += 1
                    elif isinstance(links, str):
                        link_formats["string"] += 1
                    elif isinstance(links, list):
                        link_formats["list"] += 1
                    elif isinstance(links, dict):
                        link_formats["dict"] += 1
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON decode error: {e}")
            except Exception as e:
                print(f"⚠️  Line {line_num}: Error: {e}")
    
    print(f"📊 Total examples: {total_examples}")
    print(f"✅ Valid examples: {valid_examples}")
    print(f"❌ Invalid examples: {total_examples - valid_examples}")
    
    print(f"\n📋 Document Labels Distribution:")
    for label_name, values in doc_labels.items():
        if values:
            values_array = np.array(values)
            print(f"  {label_name}:")
            print(f"    Values: {sorted(set(values))}")
            print(f"    Mean: {values_array.mean():.3f}")
            print(f"    Distribution: {Counter(values)}")
        else:
            print(f"  {label_name}: No data")
    
    print(f"\n🏷️  Span Types: {dict(span_types.most_common())}")
    print(f"\n📎 Attachment Formats: {dict(attachment_formats)}")
    print(f"\n🔗 Link Formats: {dict(link_formats)}")


def test_dataset_loading(train_path, val_path, tokenizer_path=None):
    """Test how the dataset loader processes the data"""
    print(f"\n🧪 Testing Dataset Loading")
    
    # Create tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = DLPTokenizer(tokenizer_path)
        print(f"✅ Using tokenizer: {tokenizer_path}")
    else:
        print("⚠️  Using simple fallback tokenizer")
        from scripts.train_dlp import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size=16000)
    
    # Create dataset config
    dataset_config = DLPDatasetConfig(
        max_length=1024,
        doc_labels=["sensitivity", "exposure", "context", "obfuscation"]
    )
    
    # Test loading
    try:
        print(f"\n📂 Loading training dataset: {train_path}")
        train_dataset = DLPDataset(train_path, tokenizer, dataset_config)
        print(f"✅ Training dataset loaded: {len(train_dataset)} examples")
        
        # Test first few examples
        for i in range(min(3, len(train_dataset))):
            print(f"\n🔍 Example {i}:")
            try:
                batch = train_dataset[i]
                print(f"  Input IDs shape: {batch['input_ids'].shape}")
                print(f"  Attention mask shape: {batch['attention_mask'].shape}")
                print(f"  Doc labels shape: {batch['doc_labels'].shape}")
                print(f"  Doc labels values: {batch['doc_labels']}")
                print(f"  Doc labels unique: {batch['doc_labels'].unique()}")
                print(f"  BIO labels shape: {batch['bio_labels'].shape}")
                print(f"  BIO labels unique: {batch['bio_labels'].unique()}")
                
                # Check for all-zero labels (suspicious)
                if torch.all(batch['doc_labels'] == 0):
                    print("  ⚠️  All document labels are 0!")
                if torch.all(batch['bio_labels'] == 0):
                    print("  ⚠️  All BIO labels are 0!")
                    
            except Exception as e:
                print(f"  ❌ Error loading example {i}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📂 Loading validation dataset: {val_path}")
        val_dataset = DLPDataset(val_path, tokenizer, dataset_config)
        print(f"✅ Validation dataset loaded: {len(val_dataset)} examples")
        
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🚀 HRM-DLP Data Debug Analysis")
    
    # Paths from the config
    train_path = "data/runs/run_20250824_123640_a2f52bf9/split_outputs/train_examples_augmented.jsonl"
    val_path = "data/runs/run_20250824_123640_a2f52bf9/split_outputs/val_examples_augmented.jsonl"
    tokenizer_path = "tokenizers/dlp_tokenizer.model"
    
    # Phase 1: Analyze raw data
    analyze_jsonl_file(train_path)
    analyze_jsonl_file(val_path)
    
    # Phase 2: Test dataset loading
    import torch  # Import here to avoid issues if torch not available
    test_dataset_loading(train_path, val_path, tokenizer_path)


if __name__ == "__main__":
    main()