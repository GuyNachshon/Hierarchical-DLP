#!/usr/bin/env python3
"""
Validate Generated Labels

Quick validation of the generated labels to ensure they make sense
and will enable the model to learn discrimination.
"""

import json
import numpy as np
from pathlib import Path

def validate_labels():
    print("🔍 Validating Generated Labels")
    print("=" * 50)
    
    data_file = Path("data/hrm_dlp_final/train_labeled.jsonl")
    
    # Load sample of labeled data
    examples = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Just first 20 for validation
                break
            examples.append(json.loads(line))
    
    print(f"📊 Analyzing {len(examples)} examples...")
    
    # Analyze label distributions
    labels_data = {
        'sensitivity': [ex['labels']['sensitivity'] for ex in examples],
        'exposure': [ex['labels']['exposure'] for ex in examples],  
        'context': [ex['labels']['context'] for ex in examples],
        'obfuscation': [ex['labels']['obfuscation'] for ex in examples]
    }
    
    print(f"\n📈 Label Statistics:")
    for label_type, values in labels_data.items():
        values = np.array(values)
        print(f"   {label_type.capitalize():<12}: range={values.min():.3f}-{values.max():.3f}, std={values.std():.3f}")
    
    # Show some examples
    print(f"\n📋 Sample Examples:")
    
    # Sort by total risk for variety
    examples_with_risk = []
    for ex in examples:
        labels = ex['labels']
        total_risk = sum(labels.values()) / 4
        examples_with_risk.append((ex, total_risk))
    
    examples_with_risk.sort(key=lambda x: x[1])
    
    # Show low, medium, high risk examples
    indices = [0, len(examples_with_risk)//2, -1]
    risk_levels = ["LOW", "MEDIUM", "HIGH"]
    
    for i, risk_level in zip(indices, risk_levels):
        ex, total_risk = examples_with_risk[i]
        labels = ex['labels']
        
        print(f"\n{risk_level} RISK EXAMPLE (total: {total_risk:.3f}):")
        print(f"   Subject: {ex['subject'][:60]}...")
        print(f"   Recipients: {len(ex['recipients'])} recipients")
        print(f"   Labels: S:{labels['sensitivity']:.3f} E:{labels['exposure']:.3f} C:{labels['context']:.3f} O:{labels['obfuscation']:.3f}")
        
        # Check if labeling makes sense
        subject_body = (ex['subject'] + ' ' + ex['body']).lower()
        recipients_text = ' '.join(ex['recipients']).lower()
        
        indicators = []
        if 'ssn' in subject_body or 'password' in subject_body:
            indicators.append("sensitive keywords")
        if 'gmail' in recipients_text or 'yahoo' in recipients_text:
            indicators.append("external email")
        if 'confidential' in subject_body:
            indicators.append("confidential")
        if 'urgent' in subject_body:
            indicators.append("urgent context")
            
        if indicators:
            print(f"   Risk factors: {', '.join(indicators)}")
        else:
            print(f"   Risk factors: none detected")
    
    # Check for discrimination capability
    all_values = []
    for label_type, values in labels_data.items():
        all_values.extend(values)
    
    overall_range = max(all_values) - min(all_values)
    overall_std = np.std(all_values)
    
    print(f"\n🎯 Discrimination Analysis:")
    print(f"   Overall range: {overall_range:.3f}")
    print(f"   Overall std dev: {overall_std:.3f}")
    
    if overall_range > 0.5 and overall_std > 0.15:
        print("   ✅ EXCELLENT: Strong discrimination potential")
        print("   💡 The model should learn to distinguish risk levels")
    elif overall_range > 0.3 and overall_std > 0.10:
        print("   🟢 GOOD: Adequate discrimination potential")  
        print("   💡 Model should improve significantly over baseline")
    elif overall_range > 0.1:
        print("   🟡 MODERATE: Some discrimination potential")
        print("   💡 Better than no labels, but may need tuning")
    else:
        print("   ⚠️  LIMITED: Still low discrimination")
        print("   💡 May need more aggressive labeling or different approach")
    
    print(f"\n✅ Validation complete!")
    print(f"🚀 Ready for retraining with proper labels!")

if __name__ == "__main__":
    validate_labels()