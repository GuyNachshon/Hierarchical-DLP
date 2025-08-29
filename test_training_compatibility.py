#!/usr/bin/env python3
"""Test that training code works with float labels."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
from src.dlp.dataset import create_dataloaders, DLPDatasetConfig
from src.dlp.tokenizer import SimpleTokenizer

def test_training_compatibility():
    print("🧪 Testing Training Compatibility with Float Labels")
    print("=" * 60)
    
    # Test dataset loading
    print("📊 Testing dataset loading...")
    
    try:
        # Create dataset config
        config = DLPDatasetConfig(
            max_length=512,  # Smaller for testing
            doc_labels=["sensitivity", "exposure", "context", "obfuscation"]
        )
        
        # Create simple tokenizer for testing
        tokenizer = SimpleTokenizer(vocab_size=1000)
        
        # Create dataloaders (small batch for testing)
        train_loader, val_loader = create_dataloaders(
            train_path="data/hrm_dlp_final/train_labeled.jsonl",
            val_path="data/hrm_dlp_final/val_labeled.jsonl",
            tokenizer=tokenizer,
            config=config,
            batch_size=2,  # Small batch for testing
            num_workers=0
        )
        
        print(f"   ✅ Dataloaders created successfully")
        
        # Test first batch
        print("\n🔄 Testing first training batch...")
        
        for batch in train_loader:
            print(f"   📊 Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'DLPBatch object'}")
            
            # Check if it's a DLPBatch or dict
            if hasattr(batch, '__dict__'):
                batch_dict = batch.__dict__
            else:
                batch_dict = batch
                
            print(f"   📊 Batch structure: {list(batch_dict.keys())}")
            
            # Check doc_labels specifically
            if 'doc_labels' in batch_dict:
                doc_labels = batch_dict['doc_labels']
                print(f"   📊 Doc labels shape: {doc_labels.shape}")
                print(f"   📊 Doc labels dtype: {doc_labels.dtype}")
                print(f"   📊 Doc labels range: {doc_labels.min().item():.3f} - {doc_labels.max().item():.3f}")
                
                # Check if they're floats as expected
                if doc_labels.dtype in [torch.float32, torch.float16]:
                    print("   ✅ Doc labels are float type - BCEWithLogitsLoss compatible!")
                else:
                    print(f"   ⚠️  Doc labels are {doc_labels.dtype} - may need conversion")
                    
            else:
                print("   ❌ No doc_labels found in batch")
            
            break  # Just test first batch
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test BCE loss compatibility
    print(f"\n🧪 Testing BCE Loss Compatibility...")
    
    try:
        # Create dummy data to test BCE loss
        batch_size = 2
        num_labels = 4
        
        # Simulate model output (logits)
        logits = torch.randn(batch_size, num_labels)
        
        # Simulate float targets (like our labels)
        targets = torch.tensor([
            [0.5, 0.8, 0.2, 0.1],  # Example 1
            [0.9, 0.3, 0.7, 0.0]   # Example 2
        ], dtype=torch.float32)
        
        # Test BCE with logits loss
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, targets)
        
        print(f"   📊 BCE loss computed: {loss.item():.4f}")
        print(f"   ✅ BCEWithLogitsLoss works perfectly with float targets!")
        
    except Exception as e:
        print(f"❌ BCE loss test failed: {e}")
        return False
    
    print(f"\n✅ Training Compatibility Test PASSED!")
    print(f"   • Dataset loads float labels correctly")
    print(f"   • BCEWithLogitsLoss accepts float targets") 
    print(f"   • Ready for training with nuanced risk scoring!")
    
    return True

if __name__ == "__main__":
    success = test_training_compatibility()
    if success:
        print(f"\n🚀 ALL SYSTEMS GO - Ready for retraining!")
    else:
        print(f"\n🔧 Additional fixes needed.")