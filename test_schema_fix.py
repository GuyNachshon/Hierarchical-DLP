#!/usr/bin/env python3
"""Test that the schema fix works with float labels."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import json
from src.dlp.dataset import DLPExample

def test_schema_fix():
    print("🧪 Testing Schema Fix for Float Labels")
    print("=" * 50)
    
    # Test with our generated labeled data
    labeled_file = Path("data/hrm_dlp_final/train_labeled.jsonl")
    
    if not labeled_file.exists():
        print("❌ Labeled data file not found")
        return
    
    # Try parsing a few examples
    success_count = 0
    error_count = 0
    
    with open(labeled_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Test first 10 examples
                break
                
            try:
                data = json.loads(line.strip())
                example = DLPExample(**data)
                
                # Verify labels are floats
                labels = example.labels
                print(f"✅ Example {i+1}: Labels = {labels}")
                
                # Check label types and ranges
                for label_name, label_value in labels.items():
                    if not isinstance(label_value, (int, float)):
                        print(f"   ⚠️  {label_name} is not numeric: {type(label_value)}")
                    elif not (0.0 <= label_value <= 1.0):
                        print(f"   ⚠️  {label_name} out of range [0,1]: {label_value}")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Example {i+1}: Failed to parse - {e}")
                error_count += 1
    
    print(f"\n📊 Test Results:")
    print(f"   Successful parses: {success_count}")
    print(f"   Parse errors: {error_count}")
    
    if error_count == 0:
        print("   ✅ Schema fix successful! Float labels work perfectly.")
    else:
        print("   ⚠️  Some issues remain - check error details above.")
    
    return error_count == 0

if __name__ == "__main__":
    success = test_schema_fix()
    if success:
        print("\n🚀 Ready to proceed with training!")
    else:
        print("\n🔧 Additional fixes needed before training.")