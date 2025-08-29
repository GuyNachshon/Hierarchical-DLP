#!/usr/bin/env python3
"""
HRM-DLP Data Label Diagnostic

Analyzes the training data to understand label distribution and quality.
This will help determine if the lack of model discrimination is due to data issues.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def load_jsonl_data(file_path):
    """Load JSONL data and extract relevant fields."""
    examples = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
    return examples

def analyze_labels(examples, split_name):
    """Analyze label distribution and patterns in the dataset."""
    print(f"\n{'='*60}")
    print(f"📊 LABEL ANALYSIS - {split_name.upper()} SET")
    print(f"{'='*60}")
    
    # Check if labels exist
    has_labels = any('labels' in ex for ex in examples)
    
    if not has_labels:
        print("❌ CRITICAL ISSUE: No 'labels' field found in dataset!")
        print("   The model cannot learn without ground truth labels.")
        return None
    
    # Extract labels
    labeled_examples = [ex for ex in examples if 'labels' in ex]
    unlabeled_count = len(examples) - len(labeled_examples)
    
    print(f"📋 Dataset Statistics:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Labeled examples: {len(labeled_examples)}")
    print(f"   Unlabeled examples: {unlabeled_count}")
    
    if unlabeled_count > 0:
        print(f"   ⚠️  {(unlabeled_count/len(examples))*100:.1f}% of examples lack labels!")
    
    if not labeled_examples:
        print("❌ CRITICAL ISSUE: No labeled examples found!")
        return None
    
    # Analyze label structure and values
    label_keys = set()
    for ex in labeled_examples:
        label_keys.update(ex['labels'].keys())
    
    print(f"\n📈 Label Structure:")
    print(f"   Label types found: {sorted(label_keys)}")
    
    # Expected labels for DLP
    expected_labels = {'sensitivity', 'exposure', 'context', 'obfuscation'}
    missing_labels = expected_labels - label_keys
    extra_labels = label_keys - expected_labels
    
    if missing_labels:
        print(f"   ⚠️  Missing expected labels: {missing_labels}")
    if extra_labels:
        print(f"   ℹ️  Extra labels found: {extra_labels}")
    
    # Analyze label distributions
    label_stats = {}
    
    for label_type in sorted(label_keys):
        values = []
        for ex in labeled_examples:
            if label_type in ex['labels']:
                val = ex['labels'][label_type]
                if isinstance(val, (int, float)):
                    values.append(val)
                else:
                    print(f"   ⚠️  Non-numeric label found: {label_type} = {val}")
        
        if values:
            values = np.array(values)
            label_stats[label_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'unique_values': len(np.unique(values)),
                'value_range': np.max(values) - np.min(values)
            }
    
    # Print detailed statistics
    print(f"\n📊 Label Value Analysis:")
    for label_type, stats in label_stats.items():
        print(f"   {label_type.capitalize():<12}:")
        print(f"      Range: {stats['min']:.3f} - {stats['max']:.3f} (span: {stats['value_range']:.3f})")
        print(f"      Mean:  {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"      Unique values: {stats['unique_values']}")
    
    # Check for problematic patterns
    print(f"\n🔍 Label Quality Issues:")
    issues_found = False
    
    for label_type, stats in label_stats.items():
        # Check for no variation (constant labels)
        if stats['value_range'] < 1e-6:
            print(f"   ❌ {label_type}: All values are identical ({stats['mean']:.6f})")
            issues_found = True
        
        # Check for very limited range
        elif stats['value_range'] < 0.1:
            print(f"   ⚠️  {label_type}: Very limited range ({stats['value_range']:.3f})")
            issues_found = True
        
        # Check for binary vs continuous
        if stats['unique_values'] <= 2:
            print(f"   ℹ️  {label_type}: Binary/few values ({stats['unique_values']} unique)")
        elif stats['unique_values'] > 20:
            print(f"   ℹ️  {label_type}: Many unique values ({stats['unique_values']})")
    
    if not issues_found:
        print("   ✅ No obvious label distribution issues detected")
    
    return label_stats

def analyze_content_vs_labels(examples, max_samples=20):
    """Manually examine a sample of examples to check label quality."""
    print(f"\n{'='*60}")
    print(f"🔍 CONTENT VS LABEL QUALITY CHECK")
    print(f"{'='*60}")
    
    labeled_examples = [ex for ex in examples if 'labels' in ex]
    
    if not labeled_examples:
        print("❌ No labeled examples to analyze")
        return
    
    # Sample examples across different label ranges
    sample_size = min(max_samples, len(labeled_examples))
    
    # Try to get diverse samples
    indices = np.linspace(0, len(labeled_examples)-1, sample_size, dtype=int)
    samples = [labeled_examples[i] for i in indices]
    
    print(f"📋 Analyzing {len(samples)} sample examples...\n")
    
    for i, example in enumerate(samples, 1):
        print(f"Example {i}:")
        print(f"   Subject: {example.get('subject', 'N/A')[:80]}...")
        
        # Show recipients (risk indicator)
        recipients = example.get('recipients', [])
        external_recipients = [r for r in recipients if any(domain in r for domain in ['gmail', 'yahoo', 'hotmail', 'personal'])]
        print(f"   Recipients: {len(recipients)} total, {len(external_recipients)} external")
        
        # Show attachments (risk indicator)  
        attachments = example.get('attachments', [])
        large_attachments = [a for a in attachments if a.get('size', 0) > 1000000]
        print(f"   Attachments: {len(attachments)} total, {len(large_attachments)} >1MB")
        
        # Show body snippet for PII detection
        body = example.get('body', '')
        sensitive_keywords = ['ssn', 'social security', 'password', 'credit card', 'confidential', 'api key']
        found_keywords = [kw for kw in sensitive_keywords if kw in body.lower()]
        print(f"   Sensitive keywords: {found_keywords}")
        
        # Show actual labels
        labels = example.get('labels', {})
        print(f"   Labels: S:{labels.get('sensitivity', '?'):.3f} E:{labels.get('exposure', '?'):.3f} C:{labels.get('context', '?'):.3f} O:{labels.get('obfuscation', '?'):.3f}")
        
        # Manual assessment hint
        risk_indicators = len(external_recipients) + len(large_attachments) + len(found_keywords)
        print(f"   Risk indicators: {risk_indicators} (manual assessment hint)")
        print()

def compare_splits(train_stats, val_stats, test_stats):
    """Compare label distributions across splits."""
    print(f"\n{'='*60}")
    print(f"📊 CROSS-SPLIT COMPARISON")
    print(f"{'='*60}")
    
    all_stats = {'train': train_stats, 'val': val_stats, 'test': test_stats}
    available_stats = {k: v for k, v in all_stats.items() if v is not None}
    
    if len(available_stats) < 2:
        print("⚠️  Not enough splits with labels to compare")
        return
    
    # Get common label types
    common_labels = None
    for split_stats in available_stats.values():
        if common_labels is None:
            common_labels = set(split_stats.keys())
        else:
            common_labels = common_labels.intersection(set(split_stats.keys()))
    
    if not common_labels:
        print("❌ No common labels across splits")
        return
    
    print(f"Comparing labels: {sorted(common_labels)}\n")
    
    for label_type in sorted(common_labels):
        print(f"{label_type.capitalize()}:")
        for split_name, stats in available_stats.items():
            if label_type in stats:
                s = stats[label_type]
                print(f"   {split_name:>5}: mean={s['mean']:.3f}, std={s['std']:.3f}, range={s['value_range']:.3f}")
        
        # Check for major discrepancies
        means = [stats[label_type]['mean'] for stats in available_stats.values() if label_type in stats]
        stds = [stats[label_type]['std'] for stats in available_stats.values() if label_type in stats]
        
        mean_diff = max(means) - min(means)
        std_diff = max(stds) - min(stds)
        
        if mean_diff > 0.2:
            print(f"   ⚠️  Large mean difference across splits: {mean_diff:.3f}")
        if std_diff > 0.1:
            print(f"   ⚠️  Large std difference across splits: {std_diff:.3f}")
        
        print()

def main():
    print("🔍 HRM-DLP Data Label Diagnostic")
    print("=" * 60)
    
    # Data paths
    data_dir = Path("data/hrm_dlp_final")
    
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    
    # Check file existence
    files_exist = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        files_exist[name] = path.exists()
        if path.exists():
            print(f"✅ Found {name} set: {path}")
        else:
            print(f"❌ Missing {name} set: {path}")
    
    if not any(files_exist.values()):
        print("❌ No data files found! Cannot proceed with analysis.")
        return
    
    # Load and analyze each split
    stats = {}
    
    if files_exist["train"]:
        print(f"\n🔄 Loading training data...")
        train_data = load_jsonl_data(train_path)
        print(f"   Loaded {len(train_data)} examples")
        stats['train'] = analyze_labels(train_data, "train")
        analyze_content_vs_labels(train_data, max_samples=10)
    
    if files_exist["val"]:
        print(f"\n🔄 Loading validation data...")
        val_data = load_jsonl_data(val_path)
        print(f"   Loaded {len(val_data)} examples")
        stats['val'] = analyze_labels(val_data, "val")
    
    if files_exist["test"]:
        print(f"\n🔄 Loading test data...")
        test_data = load_jsonl_data(test_path)
        print(f"   Loaded {len(test_data)} examples")
        stats['test'] = analyze_labels(test_data, "test")
    
    # Compare across splits
    compare_splits(stats.get('train'), stats.get('val'), stats.get('test'))
    
    # Final diagnosis
    print(f"\n{'='*60}")
    print(f"🎯 DIAGNOSTIC CONCLUSION")
    print(f"{'='*60}")
    
    # Check for critical issues
    critical_issues = []
    warnings = []
    
    for split_name, split_stats in stats.items():
        if split_stats is None:
            critical_issues.append(f"No labels found in {split_name} set")
            continue
            
        for label_type, label_stats in split_stats.items():
            if label_stats['value_range'] < 1e-6:
                critical_issues.append(f"{split_name} {label_type}: No variation in labels")
            elif label_stats['value_range'] < 0.1:
                warnings.append(f"{split_name} {label_type}: Very limited label range")
    
    if critical_issues:
        print("❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"   • {issue}")
        print("\n💡 LIKELY CAUSE: Training/Data Problem")
        print("   The model cannot learn discrimination without varied labels!")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Check data generation pipeline - are labels being computed correctly?")
        print("   2. Verify ground truth labeling logic")
        print("   3. Add examples with obvious high/low risk patterns")
        print("   4. Consider synthetic label generation for testing")
        
    elif warnings:
        print("⚠️  POTENTIAL ISSUES FOUND:")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n💡 LIKELY CAUSE: Training/Calibration Problem")
        print("   Labels may be too conservative or not well-distributed")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Increase label range/variance in data generation")
        print("   2. Add extreme examples (very high and very low risk)")
        print("   3. Check loss function weighting")
        
    else:
        print("✅ LABEL DISTRIBUTION LOOKS REASONABLE")
        print("   The discrimination issue is likely in training or model architecture")
        
        print("\n🔧 NEXT DIAGNOSTIC STEPS:")
        print("   1. Check training loss function and weights")
        print("   2. Monitor training dynamics (does model ever discriminate?)")
        print("   3. Test model architecture with synthetic data")

if __name__ == "__main__":
    main()