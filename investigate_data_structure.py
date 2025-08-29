#!/usr/bin/env python3
"""
Investigate the actual data structure to understand what we have
and what's missing for DLP training.
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_data_structure(file_path, max_examples=5):
    """Analyze the structure of JSONL data."""
    print(f"\n🔍 Analyzing: {file_path}")
    
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError:
                continue
    
    if not examples:
        print("❌ No valid examples found")
        return
    
    print(f"📊 Loaded {len(examples)} examples for analysis")
    
    # Analyze structure
    all_keys = set()
    for example in examples:
        all_keys.update(example.keys())
    
    print(f"\n📋 Top-level fields found: {sorted(all_keys)}")
    
    # Check each field type and content
    for key in sorted(all_keys):
        values = [example.get(key) for example in examples if key in example]
        
        print(f"\n🔍 Field: {key}")
        print(f"   Present in: {len(values)}/{len(examples)} examples")
        
        if values:
            first_val = values[0]
            if isinstance(first_val, (str, int, float)):
                print(f"   Type: {type(first_val).__name__}")
                if isinstance(first_val, str) and len(first_val) > 100:
                    print(f"   Sample: {first_val[:100]}...")
                else:
                    print(f"   Sample: {first_val}")
            elif isinstance(first_val, list):
                print(f"   Type: list (length {len(first_val)})")
                if first_val:
                    print(f"   Sample item: {first_val[0]}")
            elif isinstance(first_val, dict):
                print(f"   Type: dict with keys: {list(first_val.keys())}")
                # Special handling for metadata and attachments
                if key == '_metadata':
                    print(f"   Metadata sample: {first_val}")
                elif key == 'attachments':
                    if first_val:
                        att = first_val[0]
                        print(f"   Attachment keys: {list(att.keys())}")
                        if 'sensitivity_indicators' in att:
                            print(f"   Sensitivity indicators: {att['sensitivity_indicators']}")
    
    # Look for any label-like fields
    print(f"\n🎯 Looking for potential label fields...")
    label_candidates = []
    
    for example in examples:
        for key, value in example.items():
            if 'label' in key.lower() or 'score' in key.lower() or 'risk' in key.lower():
                label_candidates.append((key, type(value).__name__, value))
    
    if label_candidates:
        print("   Found potential label fields:")
        for key, type_name, value in label_candidates[:10]:  # Limit output
            print(f"      {key} ({type_name}): {value}")
    else:
        print("   ❌ No obvious label fields found")
    
    # Check sensitivity indicators in attachments
    print(f"\n🔍 Analyzing sensitivity indicators...")
    all_indicators = set()
    for example in examples:
        for att in example.get('attachments', []):
            indicators = att.get('sensitivity_indicators', [])
            all_indicators.update(indicators)
    
    if all_indicators:
        print(f"   Found sensitivity indicators: {sorted(all_indicators)}")
        print("   💡 These could be used to generate labels!")
    else:
        print("   No sensitivity indicators found")
    
    return examples

def check_training_expectations():
    """Check what the training code expects vs what we have."""
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING EXPECTATION VS REALITY")
    print(f"{'='*60}")
    
    print("Expected by training code:")
    print("   • 'labels' field with 4 scores:")
    print("     - sensitivity: float [0,1]")
    print("     - exposure: float [0,1]") 
    print("     - context: float [0,1]")
    print("     - obfuscation: float [0,1]")
    
    print("\nWhat we actually have:")
    print("   • Rich content (subject, body, recipients, attachments)")
    print("   • Metadata (_metadata with model info)")
    print("   • Sensitivity indicators in attachments")
    print("   • ❌ NO explicit numeric labels")
    
    print("\n💡 SOLUTION OPTIONS:")
    print("   1. Generate labels from existing content (rule-based)")
    print("   2. Use LLM to label existing data") 
    print("   3. Create synthetic labeled data")
    print("   4. Use sensitivity indicators to derive scores")

def main():
    print("🔍 HRM-DLP Data Structure Investigation")
    print("=" * 60)
    
    data_dir = Path("data/hrm_dlp_final")
    
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if file_path.exists():
            analyze_data_structure(file_path, max_examples=3)
    
    check_training_expectations()

if __name__ == "__main__":
    main()