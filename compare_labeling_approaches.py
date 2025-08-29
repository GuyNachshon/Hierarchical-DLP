#!/usr/bin/env python3
"""
Compare Pattern-Based vs Contextual Labeling Approaches

Demonstrates the strategic difference between competing with regex
vs complementing it with semantic understanding.
"""

import json
import numpy as np
from pathlib import Path

def load_comparison_data(file_path: Path, max_examples: int = 50):
    """Load examples with both labeling approaches."""
    examples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            try:
                example = json.loads(line)
                if 'labels' in example and 'contextual_labels' in example:
                    examples.append(example)
            except:
                continue
    
    return examples

def analyze_labeling_philosophy():
    """Show the core philosophical differences between approaches."""
    print("🎯 LABELING PHILOSOPHY COMPARISON")
    print("=" * 60)
    
    print("📊 PATTERN-BASED APPROACH (Competing with Regex):")
    print("   Focus: Detecting sensitive patterns in content")
    print("   Strengths: Good at finding PII, passwords, credit cards")  
    print("   Weaknesses: Misses context, business appropriateness")
    print("   Use case: 'Find all SSNs and credit card numbers'")
    print("   Problem: Regex already does this better!")
    
    print("\n🧠 CONTEXTUAL APPROACH (Complementing Regex):")
    print("   Focus: Understanding business context and appropriateness")
    print("   Strengths: Role-based analysis, intent detection, business logic")
    print("   Weaknesses: May miss obvious patterns (but regex catches those)")
    print("   Use case: 'Is this sharing appropriate for this user/situation?'")
    print("   Advantage: Does what regex CAN'T do!")

def find_strategic_examples(examples):
    """Find examples that highlight the strategic difference."""
    print(f"\n🔍 STRATEGIC EXAMPLE ANALYSIS")
    print("=" * 60)
    
    strategic_cases = []
    
    for example in examples:
        pattern_labels = {k: v for k, v in example['labels'].items() 
                         if k in ['sensitivity', 'exposure', 'context', 'obfuscation']}
        contextual_labels = example['contextual_labels']
        
        # Calculate total risk scores
        pattern_risk = sum(pattern_labels.values()) / 4 if pattern_labels else 0
        contextual_risk = sum(contextual_labels.values()) / 4
        
        # Find cases where approaches disagree significantly
        risk_diff = abs(pattern_risk - contextual_risk)
        
        if risk_diff > 0.2:  # Significant disagreement
            strategic_cases.append({
                'example': example,
                'pattern_risk': pattern_risk,
                'contextual_risk': contextual_risk,
                'difference': risk_diff,
                'pattern_labels': pattern_labels,
                'contextual_labels': contextual_labels
            })
    
    # Sort by difference and show top cases
    strategic_cases.sort(key=lambda x: x['difference'], reverse=True)
    
    print(f"Found {len(strategic_cases)} cases where approaches disagree significantly\n")
    
    for i, case in enumerate(strategic_cases[:5]):
        example = case['example']
        print(f"📧 STRATEGIC CASE {i+1}:")
        print(f"   Subject: {example['subject'][:60]}...")
        print(f"   Recipients: {example.get('recipients', [])}")
        
        print(f"\n   🤖 PATTERN-BASED RISK: {case['pattern_risk']:.3f}")
        p_labels = case['pattern_labels']
        print(f"      S:{p_labels.get('sensitivity', 0):.3f} E:{p_labels.get('exposure', 0):.3f} C:{p_labels.get('context', 0):.3f} O:{p_labels.get('obfuscation', 0):.3f}")
        
        print(f"   🧠 CONTEXTUAL RISK: {case['contextual_risk']:.3f}")
        c_labels = case['contextual_labels']
        print(f"      S:{c_labels['sensitivity']:.3f} E:{c_labels['exposure']:.3f} C:{c_labels['context']:.3f} O:{c_labels['obfuscation']:.3f}")
        
        # Analyze why they differ
        print(f"   📊 DIFFERENCE: {case['difference']:.3f}")
        
        content = (example['subject'] + ' ' + example['body']).lower()
        recipients = ' '.join(example.get('recipients', [])).lower()
        
        pattern_indicators = []
        if any(word in content for word in ['ssn', 'password', 'credit', 'confidential']):
            pattern_indicators.append("sensitive keywords detected")
        if any(domain in recipients for domain in ['gmail', 'yahoo', 'hotmail']):
            pattern_indicators.append("personal email domains")
            
        contextual_indicators = []
        if 'urgent' in content:
            contextual_indicators.append("pressure tactics")
        if 'board' in content or 'meeting' in content:
            contextual_indicators.append("business context")
        if len(example.get('recipients', [])) > 3:
            contextual_indicators.append("high recipient count")
            
        print(f"   🔍 Pattern signals: {pattern_indicators or ['none detected']}")
        print(f"   🔍 Context signals: {contextual_indicators or ['none detected']}")
        print()

def calculate_approach_statistics(examples):
    """Calculate overall statistics for both approaches."""
    print(f"📈 OVERALL APPROACH STATISTICS")
    print("=" * 60)
    
    pattern_risks = []
    contextual_risks = []
    
    for example in examples:
        pattern_labels = {k: v for k, v in example['labels'].items() 
                         if k in ['sensitivity', 'exposure', 'context', 'obfuscation']}
        contextual_labels = example['contextual_labels']
        
        if pattern_labels:
            pattern_risk = sum(pattern_labels.values()) / 4
            pattern_risks.append(pattern_risk)
        
        contextual_risk = sum(contextual_labels.values()) / 4
        contextual_risks.append(contextual_risk)
    
    if pattern_risks and contextual_risks:
        print(f"📊 PATTERN-BASED APPROACH:")
        p_array = np.array(pattern_risks)
        print(f"   Mean risk: {p_array.mean():.3f}")
        print(f"   Std dev:   {p_array.std():.3f}")
        print(f"   Range:     {p_array.min():.3f} - {p_array.max():.3f}")
        
        print(f"\n📊 CONTEXTUAL APPROACH:")
        c_array = np.array(contextual_risks)
        print(f"   Mean risk: {c_array.mean():.3f}")
        print(f"   Std dev:   {c_array.std():.3f}") 
        print(f"   Range:     {c_array.min():.3f} - {c_array.max():.3f}")
        
        # Correlation analysis
        if len(pattern_risks) == len(contextual_risks):
            correlation = np.corrcoef(pattern_risks, contextual_risks)[0,1]
            print(f"\n📊 APPROACH CORRELATION: {correlation:.3f}")
            
            if correlation < 0.3:
                print("   ✅ LOW CORRELATION - Approaches are complementary!")
                print("   💡 They focus on different risk dimensions")
            elif correlation < 0.6:
                print("   🟡 MEDIUM CORRELATION - Some overlap but still valuable")
            else:
                print("   ⚠️  HIGH CORRELATION - Approaches are redundant")

def recommend_deployment_strategy():
    """Provide recommendations for production deployment."""
    print(f"\n🚀 PRODUCTION DEPLOYMENT RECOMMENDATIONS")
    print("=" * 60)
    
    print("🏗️  HYBRID ARCHITECTURE:")
    print("   1. REGEX LAYER (Fast, Pattern-Based)")
    print("      • Detect obvious PII: SSNs, credit cards, phone numbers")
    print("      • Flag sensitive keywords: 'confidential', 'secret'")  
    print("      • Block obvious violations immediately")
    print("      • Low latency, high throughput")
    
    print("\n   2. ML LAYER (Contextual Understanding)")
    print("      • Analyze business appropriateness")
    print("      • Consider user roles and relationships")
    print("      • Detect intent and social engineering")
    print("      • Handle edge cases regex misses")
    
    print("\n📊 PROCESSING FLOW:")
    print("   Email → Regex Check → ML Context Analysis → Final Decision")
    print("   Fast reject ↗️        ↘️ Nuanced evaluation")
    
    print("\n🎯 EXPECTED BENEFITS:")
    print("   ✅ Complementary strengths, not competing systems")
    print("   ✅ Regex handles obvious cases (fast)")
    print("   ✅ ML handles complex business context (accurate)")
    print("   ✅ Better overall coverage than either alone")
    print("   ✅ Reduced false positives from pure pattern matching")

def main():
    print("🔄 Pattern-Based vs Contextual Labeling Comparison")
    print("=" * 60)
    
    # Load comparison data
    data_file = Path("data/hrm_dlp_final/train_contextual.jsonl")
    
    if not data_file.exists():
        print("❌ Contextual data not found. Run: python generate_contextual_labels.py")
        return
    
    examples = load_comparison_data(data_file, max_examples=100)
    
    if not examples:
        print("❌ No examples with both label types found")
        return
    
    print(f"📊 Loaded {len(examples)} examples for comparison\n")
    
    # Run analysis
    analyze_labeling_philosophy()
    find_strategic_examples(examples)
    calculate_approach_statistics(examples)
    recommend_deployment_strategy()
    
    print(f"\n✅ CONCLUSION:")
    print(f"   The contextual approach creates a model that COMPLEMENTS")
    print(f"   rather than COMPETES with regex-based DLP systems.")
    print(f"   This provides strategic value and avoids redundancy!")

if __name__ == "__main__":
    main()