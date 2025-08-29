#!/usr/bin/env python3
"""
Generate Contextual DLP Labels

Creates labels focused on business context and semantic understanding
rather than pattern matching. This creates a model that complements
regex-based DLP systems.
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from contextual_label_strategy import ContextualDLPLabeler
import numpy as np

def process_dataset_contextual(input_path: Path, output_path: Path, labeler: ContextualDLPLabeler):
    """Process dataset with contextual labeling approach."""
    print(f"🔄 Processing: {input_path}")
    
    processed_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                example = json.loads(line.strip())
                
                # Generate contextual labels
                contextual_labels = labeler.generate_contextual_labels(example)
                
                # Add both old and new labels for comparison
                if 'labels' not in example:
                    example['labels'] = {}
                
                example['labels'].update(contextual_labels)
                example['contextual_labels'] = contextual_labels
                
                # Write back to file
                outfile.write(json.dumps(example) + '\n')
                processed_count += 1
                
                # Progress indicator
                if line_num % 100 == 0:
                    print(f"   Processed {line_num} examples...")
                    
            except json.JSONDecodeError:
                print(f"   ⚠️  Skipping malformed line {line_num}")
                continue
            except Exception as e:
                print(f"   ❌ Error processing line {line_num}: {e}")
                continue
    
    print(f"   ✅ Successfully processed {processed_count} examples")
    return processed_count

def analyze_label_differences(file_path: Path, sample_size: int = 20):
    """Compare pattern-based vs contextual labels on sample data."""
    print(f"\n🔍 Analyzing label differences in {file_path.name}...")
    
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                example = json.loads(line)
                if 'labels' in example and 'contextual_labels' in example:
                    examples.append(example)
            except:
                continue
    
    if not examples:
        print("   No examples with both label types found")
        return
    
    # Calculate differences
    differences = []
    for example in examples:
        pattern_labels = {k: v for k, v in example['labels'].items() if k not in example['contextual_labels']}
        contextual_labels = example['contextual_labels']
        
        if len(pattern_labels) == 4:  # Has both types
            diff_scores = {}
            for label_type in ['sensitivity', 'exposure', 'context', 'obfuscation']:
                if label_type in pattern_labels and label_type in contextual_labels:
                    diff = abs(pattern_labels[label_type] - contextual_labels[label_type])
                    diff_scores[label_type] = diff
            differences.append(diff_scores)
    
    if differences:
        # Calculate average differences
        avg_diffs = {}
        for label_type in ['sensitivity', 'exposure', 'context', 'obfuscation']:
            diffs = [d.get(label_type, 0) for d in differences]
            avg_diffs[label_type] = np.mean(diffs) if diffs else 0
        
        print(f"   Average labeling differences:")
        for label_type, avg_diff in avg_diffs.items():
            print(f"      {label_type.capitalize():<12}: {avg_diff:.3f}")
        
        print(f"   Overall difference: {np.mean(list(avg_diffs.values())):.3f}")
    
    # Show some examples where approaches differ significantly
    print(f"\n📋 Examples where approaches differ most:")
    
    example_diffs = []
    for i, example in enumerate(examples[:10]):
        if 'contextual_labels' in example:
            contextual = example['contextual_labels']
            
            # Calculate total risk scores
            contextual_risk = sum(contextual.values()) / 4
            
            example_diffs.append((i, example, contextual_risk))
    
    # Sort by risk difference and show interesting cases
    example_diffs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (idx, example, contextual_risk) in enumerate(example_diffs[:3]):
        print(f"\n   Example {i+1}: {example['subject'][:50]}...")
        print(f"      User: {example.get('user', {}).get('role', 'Unknown')} from {example.get('user', {}).get('dept', 'Unknown')}")
        print(f"      Recipients: {len(example.get('recipients', []))} recipients")
        
        contextual = example['contextual_labels']
        print(f"      Contextual: S:{contextual['sensitivity']:.3f} E:{contextual['exposure']:.3f} C:{contextual['context']:.3f} O:{contextual['obfuscation']:.3f}")
        print(f"      Contextual Risk: {contextual_risk:.3f}")

def main():
    print("🧠 Contextual DLP Label Generation")
    print("=" * 50)
    print("Focus: Business context and semantic understanding")
    print("Complements: Regex-based pattern detection")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    labeler = ContextualDLPLabeler()
    
    # Process each split
    data_dir = Path("data/hrm_dlp_final")
    
    for split in ["train", "val", "test"]:
        input_path = data_dir / f"{split}_labeled.jsonl"  # Use existing labeled data
        output_path = data_dir / f"{split}_contextual.jsonl"
        
        if not input_path.exists():
            print(f"⚠️  Skipping {split}: file not found")
            continue
        
        print(f"\n📋 Processing {split} set with contextual approach...")
        processed = process_dataset_contextual(input_path, output_path, labeler)
        
        if processed > 0:
            print(f"   💾 Saved contextual labels to: {output_path}")
            
            # Analyze differences between approaches
            analyze_label_differences(output_path)
    
    print(f"\n✅ Contextual label generation complete!")
    print(f"\n🎯 Key Differences from Pattern-Based Approach:")
    print(f"   • Focus on business appropriateness vs pattern detection")
    print(f"   • User role/department context matters more than keywords")
    print(f"   • Recipient relationships analyzed for business justification")
    print(f"   • Intent analysis vs simple content scanning")
    
    print(f"\n🔄 Next Steps:")
    print(f"   1. Compare model performance: Pattern vs Contextual labels")
    print(f"   2. Train model with contextual labels: use *_contextual.jsonl")
    print(f"   3. Test on business scenarios regex can't handle")
    print(f"   4. Validate complementarity with existing DLP tools")

if __name__ == "__main__":
    main()