#!/usr/bin/env python3
"""
Compare LLM vs Rule-Based Labeling Approaches

Analyzes and compares the quality, consistency, and strategic focus
of LLM-generated labels vs rule-based heuristic labels.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
# Optional visualization imports - not required for core functionality
# import matplotlib.pyplot as plt
# import seaborn as sns

class LabelingComparisonAnalyzer:
    """Analyzer for comparing different labeling approaches."""
    
    def __init__(self):
        self.comparison_results = {}
        
    def load_labeled_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load labeled data from JSONL file."""
        examples = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num}")
                    continue
        
        return examples
    
    def extract_labels(self, examples: List[Dict[str, Any]], 
                      label_key: str = 'labels') -> Dict[str, List[float]]:
        """Extract labels from examples."""
        labels_data = defaultdict(list)
        
        for example in examples:
            if label_key in example and isinstance(example[label_key], dict):
                for label_type, value in example[label_key].items():
                    if isinstance(value, (int, float)):
                        labels_data[label_type].append(float(value))
        
        return dict(labels_data)
    
    def compute_label_statistics(self, labels_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for labels."""
        stats = {}
        
        for label_type, values in labels_data.items():
            if values:
                values_array = np.array(values)
                stats[label_type] = {
                    'mean': float(values_array.mean()),
                    'std': float(values_array.std()),
                    'min': float(values_array.min()),
                    'max': float(values_array.max()),
                    'range': float(values_array.max() - values_array.min()),
                    'count': len(values)
                }
        
        return stats
    
    def compare_label_distributions(self, rule_labels: Dict[str, List[float]], 
                                  llm_labels: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare distributions between rule-based and LLM labels."""
        comparison = {}
        
        # Common label types
        common_types = set(rule_labels.keys()) & set(llm_labels.keys())
        
        for label_type in common_types:
            rule_vals = np.array(rule_labels[label_type])
            llm_vals = np.array(llm_labels[label_type])
            
            # Ensure same length for correlation
            min_len = min(len(rule_vals), len(llm_vals))
            rule_vals = rule_vals[:min_len]
            llm_vals = llm_vals[:min_len]
            
            # Compute comparison metrics
            correlation = np.corrcoef(rule_vals, llm_vals)[0, 1] if min_len > 1 else 0.0
            mean_diff = np.mean(llm_vals) - np.mean(rule_vals)
            std_diff = np.std(llm_vals) - np.std(rule_vals)
            range_diff = (np.max(llm_vals) - np.min(llm_vals)) - (np.max(rule_vals) - np.min(rule_vals))
            
            comparison[label_type] = {
                'correlation': float(correlation),
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff), 
                'range_difference': float(range_diff),
                'rule_stats': {
                    'mean': float(np.mean(rule_vals)),
                    'std': float(np.std(rule_vals)),
                    'range': float(np.max(rule_vals) - np.min(rule_vals))
                },
                'llm_stats': {
                    'mean': float(np.mean(llm_vals)),
                    'std': float(np.std(llm_vals)),
                    'range': float(np.max(llm_vals) - np.min(llm_vals))
                }
            }
        
        return comparison
    
    def analyze_contextual_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find examples where LLM and rule-based approaches differ significantly."""
        contextual_cases = []
        
        for i, example in enumerate(examples):
            if 'labels' not in example or 'llm_labels' not in example:
                continue
                
            rule_labels = example['labels']
            llm_labels = example['llm_labels']
            
            # Calculate overall risk scores
            rule_risk = sum(rule_labels.values()) / len(rule_labels)
            llm_risk = sum(llm_labels.values()) / len(llm_labels)
            
            risk_diff = abs(rule_risk - llm_risk)
            
            # Find cases with significant disagreement
            if risk_diff > 0.2:
                # Analyze why they might differ
                content = (example.get('subject', '') + ' ' + example.get('body', '')).lower()
                recipients = example.get('recipients', [])
                user = example.get('user', {})
                
                contextual_factors = []
                
                # Check for business context indicators
                if any(term in content for term in ['board', 'meeting', 'review', 'preparation']):
                    contextual_factors.append('Business context present')
                
                # Check for role-recipient appropriateness
                if user.get('role') in ['CFO', 'CEO', 'DIRECTOR'] and '@company.com' in str(recipients):
                    contextual_factors.append('Senior role + internal recipients')
                elif user.get('role') in ['INTERN', 'TEMP'] and any('@gmail' in r or '@yahoo' in r for r in recipients):
                    contextual_factors.append('Junior role + external recipients')
                
                # Check for pressure/manipulation indicators
                if any(term in content for term in ['urgent', 'immediately', 'secret', 'confidential']):
                    contextual_factors.append('Pressure/confidentiality language')
                
                contextual_cases.append({
                    'index': i,
                    'subject': example.get('subject', ''),
                    'user': user,
                    'recipients': recipients,
                    'rule_risk': rule_risk,
                    'llm_risk': llm_risk,
                    'risk_difference': risk_diff,
                    'rule_labels': rule_labels,
                    'llm_labels': llm_labels,
                    'contextual_factors': contextual_factors,
                    'llm_reasoning': example.get('llm_reasoning', {})
                })
        
        # Sort by risk difference (most disagreement first)
        contextual_cases.sort(key=lambda x: x['risk_difference'], reverse=True)
        
        return contextual_cases
    
    def generate_comparison_report(self, rule_file: Path, llm_file: Path, 
                                 output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        
        print(f"📊 Labeling Approach Comparison Report")
        print("=" * 60)
        
        # Load data
        print(f"📁 Loading rule-based labels: {rule_file.name}")
        rule_examples = self.load_labeled_data(rule_file)
        rule_labels = self.extract_labels(rule_examples, 'labels')
        
        print(f"📁 Loading LLM labels: {llm_file.name}")
        llm_examples = self.load_labeled_data(llm_file)
        llm_labels = self.extract_labels(llm_examples, 'llm_labels')
        
        if not rule_labels or not llm_labels:
            print("❌ Failed to extract labels from one or both files")
            return {}
        
        print(f"✅ Loaded {len(rule_examples)} rule-based and {len(llm_examples)} LLM examples")
        
        # Compute statistics
        rule_stats = self.compute_label_statistics(rule_labels)
        llm_stats = self.compute_label_statistics(llm_labels)
        
        # Compare distributions
        distribution_comparison = self.compare_label_distributions(rule_labels, llm_labels)
        
        # Find contextual examples (assumes same examples with both label types)
        if len(llm_examples) > 0 and 'labels' in llm_examples[0] and 'llm_labels' in llm_examples[0]:
            contextual_cases = self.analyze_contextual_examples(llm_examples)
        else:
            contextual_cases = []
        
        # Generate report
        report = {
            'summary': {
                'rule_based_examples': len(rule_examples),
                'llm_examples': len(llm_examples),
                'comparison_date': np.datetime64('now').astype(str)
            },
            'rule_based_stats': rule_stats,
            'llm_stats': llm_stats,
            'distribution_comparison': distribution_comparison,
            'contextual_analysis': {
                'disagreement_cases': len(contextual_cases),
                'top_differences': contextual_cases[:5]  # Top 5 cases
            }
        }
        
        # Print key findings
        self.print_comparison_summary(report)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Detailed report saved: {output_file}")
        
        return report
    
    def print_comparison_summary(self, report: Dict[str, Any]):
        """Print summary of comparison results."""
        print(f"\n📈 COMPARISON SUMMARY")
        print("=" * 40)
        
        # Distribution comparison
        dist_comp = report['distribution_comparison']
        
        print(f"📊 Label Distribution Comparison:")
        for label_type, metrics in dist_comp.items():
            correlation = metrics['correlation']
            mean_diff = metrics['mean_difference']
            range_diff = metrics['range_difference']
            
            print(f"   {label_type.capitalize():<12}:")
            print(f"      Correlation:     {correlation:.3f}")
            print(f"      Mean difference: {mean_diff:+.3f}")
            print(f"      Range difference:{range_diff:+.3f}")
        
        # Overall assessment
        avg_correlation = np.mean([m['correlation'] for m in dist_comp.values()])
        
        print(f"\n🎯 Overall Assessment:")
        print(f"   Average correlation: {avg_correlation:.3f}")
        
        if avg_correlation < 0.3:
            print("   ✅ LOW CORRELATION - Approaches are complementary!")
            print("   💡 LLM focuses on different risk dimensions than rules")
        elif avg_correlation < 0.6:
            print("   🟡 MEDIUM CORRELATION - Some overlap but valuable differences")
            print("   💡 LLM provides additional insights beyond rule-based")
        else:
            print("   ⚠️  HIGH CORRELATION - Approaches are similar")
            print("   💡 Consider focusing LLM on more contextual factors")
        
        # Contextual analysis
        contextual = report['contextual_analysis']
        
        if contextual['disagreement_cases'] > 0:
            print(f"\n🔍 Contextual Differences:")
            print(f"   {contextual['disagreement_cases']} cases with significant disagreement")
            
            for i, case in enumerate(contextual['top_differences'][:3], 1):
                print(f"\n   Example {i}: {case['subject'][:50]}...")
                print(f"      Rule risk: {case['rule_risk']:.3f}, LLM risk: {case['llm_risk']:.3f}")
                print(f"      Factors: {', '.join(case['contextual_factors'])}")
                
                if case['llm_reasoning']:
                    print(f"      LLM reasoning: {case['llm_reasoning'].get('sensitivity', 'N/A')}")
        
        print(f"\n💡 STRATEGIC INSIGHTS:")
        
        # Calculate discrimination metrics
        rule_ranges = [m['rule_stats']['range'] for m in dist_comp.values()]
        llm_ranges = [m['llm_stats']['range'] for m in dist_comp.values()]
        
        avg_rule_range = np.mean(rule_ranges)
        avg_llm_range = np.mean(llm_ranges)
        
        if avg_llm_range > avg_rule_range:
            print("   ✅ LLM shows better discrimination than rule-based approach")
            print("   ✅ LLM labels span wider range - better for training")
        else:
            print("   🟡 Rule-based shows comparable discrimination")
            print("   💡 Consider tuning LLM prompts for better discrimination")
        
        if avg_correlation < 0.5:
            print("   ✅ LLM approach truly complements rule-based systems")
            print("   ✅ Strategic goal achieved: focus on context vs patterns")

def create_synthetic_comparison():
    """Create synthetic comparison to demonstrate the tool."""
    print("🧪 Creating Synthetic Comparison Demo")
    print("=" * 50)
    
    # Create sample data with both types of labels
    synthetic_data = [
        {
            "subject": "Board meeting financial summary",
            "user": {"role": "CFO", "dept": "FINANCE"},
            "recipients": ["board@company.com"],
            "body": "Q4 results for board review...",
            "labels": {"sensitivity": 0.8, "exposure": 0.3, "context": 0.2, "obfuscation": 0.1},
            "llm_labels": {"sensitivity": 0.3, "exposure": 0.1, "context": 0.1, "obfuscation": 0.0},
            "llm_reasoning": {
                "sensitivity": "Low - appropriate business context for CFO sharing with board",
                "exposure": "Very low - appropriate recipients for financial data",
                "context": "Very low - perfect role-based access",
                "obfuscation": "None - transparent business communication"
            }
        },
        {
            "subject": "Found some interesting salary data!",
            "user": {"role": "INTERN", "dept": "MARKETING"},
            "recipients": ["friend@gmail.com"],
            "body": "Check out these executive salaries...",
            "labels": {"sensitivity": 0.6, "exposure": 0.4, "context": 0.2, "obfuscation": 0.1},
            "llm_labels": {"sensitivity": 0.9, "exposure": 0.8, "context": 0.9, "obfuscation": 0.3},
            "llm_reasoning": {
                "sensitivity": "Very high - inappropriate sharing of confidential salary data",
                "exposure": "High - personal Gmail account inappropriate for business data",
                "context": "Very high - intern has no justification for accessing salary data", 
                "obfuscation": "Moderate - casual tone suggests lack of awareness"
            }
        }
    ]
    
    # Save synthetic data
    synthetic_file = Path("synthetic_comparison_demo.jsonl")
    with open(synthetic_file, 'w') as f:
        for example in synthetic_data:
            f.write(json.dumps(example) + '\n')
    
    # Create analyzer and run comparison
    analyzer = LabelingComparisonAnalyzer()
    
    # For demo, use the same file for both (contains both label types)
    report = analyzer.generate_comparison_report(
        synthetic_file, synthetic_file, 
        Path("comparison_report_demo.json")
    )
    
    # Cleanup
    synthetic_file.unlink()
    
    return report

def main():
    """Main function for comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare LLM vs Rule-based labeling approaches")
    parser.add_argument("--rule_file", type=str, 
                       help="JSONL file with rule-based labels")
    parser.add_argument("--llm_file", type=str,
                       help="JSONL file with LLM labels")
    parser.add_argument("--output", type=str,
                       help="Output file for detailed report")
    parser.add_argument("--demo", action="store_true",
                       help="Run synthetic comparison demo")
    
    args = parser.parse_args()
    
    if args.demo:
        create_synthetic_comparison()
        return
    
    if not args.rule_file or not args.llm_file:
        print("❌ Both --rule_file and --llm_file are required")
        print("   Or use --demo to see a synthetic comparison")
        return
    
    rule_file = Path(args.rule_file)
    llm_file = Path(args.llm_file)
    
    if not rule_file.exists():
        print(f"❌ Rule-based file not found: {rule_file}")
        return
    
    if not llm_file.exists():
        print(f"❌ LLM file not found: {llm_file}")
        return
    
    output_file = Path(args.output) if args.output else None
    
    analyzer = LabelingComparisonAnalyzer()
    report = analyzer.generate_comparison_report(rule_file, llm_file, output_file)

if __name__ == "__main__":
    main()