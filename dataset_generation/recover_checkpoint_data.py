#!/usr/bin/env python3
"""
Recovery script to properly save checkpoint data to final output files.

This fixes the issue where 334 examples are in checkpoints but only 53 in final output.
"""

import json
import os
from pathlib import Path

def recover_checkpoint_data():
    """Recover data from checkpoint file and create proper final outputs"""
    
    checkpoint_file = "/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP/HRM/scripts/data/dlp_agentic/.checkpoints/completed_examples.jsonl"
    output_dir = "/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP/data/dlp_agentic"
    
    print(f"🔄 Recovering checkpoint data from: {checkpoint_file}")
    
    # Load all examples from checkpoint
    examples = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping malformed line: {e}")
    
    print(f"📊 Loaded {len(examples)} examples from checkpoint")
    
    if not examples:
        print("❌ No examples found in checkpoint file")
        return
    
    # Create proper train/val/test splits
    # Use 70/15/15 split approximately
    train_size = int(len(examples) * 0.7)  # ~234 examples
    val_size = int(len(examples) * 0.15)   # ~50 examples  
    test_size = len(examples) - train_size - val_size  # remaining ~50
    
    splits = [
        ("train", examples[:train_size]),
        ("val", examples[train_size:train_size + val_size]),
        ("test", examples[train_size + val_size:])
    ]
    
    # Save each split
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_examples in splits:
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        
        print(f"💾 Saving {len(split_examples)} examples to {split_name}.jsonl...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in split_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"✅ {split_name}.jsonl: {len(split_examples)} examples saved")
    
    # Create recovery stats
    stats = {
        "recovery_info": {
            "source": "checkpoint file recovery", 
            "total_recovered": len(examples),
            "splits_created": {
                "train": train_size,
                "val": val_size, 
                "test": test_size
            }
        }
    }
    
    stats_file = os.path.join(output_dir, "recovery_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n🎉 Recovery completed!")
    print(f"📊 Total recovered: {len(examples)} examples")
    print(f"📁 Train: {train_size} examples")
    print(f"📁 Val: {val_size} examples") 
    print(f"📁 Test: {test_size} examples")
    print(f"📋 Recovery stats saved to: {stats_file}")

if __name__ == "__main__":
    recover_checkpoint_data()