#!/usr/bin/env python3
"""
HRM-DLP Benchmark Test

Evaluates model performance on the test dataset and computes metrics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
sys.path.append(str(Path(__file__).parent / "src"))

from test_model import HRMDLPTester

def compute_metrics(predictions, ground_truth, threshold=0.5):
    """Compute classification metrics."""
    metrics = {}
    
    # Document-level metrics
    doc_labels = ['sensitivity', 'exposure', 'context', 'obfuscation']
    
    for i, label in enumerate(doc_labels):
        if label in ground_truth and len(ground_truth[label]) > 0:
            y_true = np.array(ground_truth[label])
            y_pred_prob = np.array(predictions[label])
            y_pred = (y_pred_prob > threshold).astype(int)
            
            # Binary classification metrics
            try:
                auc = roc_auc_score(y_true, y_pred_prob)
                precision = np.mean((y_pred == 1) & (y_true == 1)) / max(np.mean(y_pred == 1), 1e-8)
                recall = np.mean((y_pred == 1) & (y_true == 1)) / max(np.mean(y_true == 1), 1e-8)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                
                metrics[f'{label}_auc'] = auc
                metrics[f'{label}_precision'] = precision
                metrics[f'{label}_recall'] = recall
                metrics[f'{label}_f1'] = f1
                metrics[f'{label}_accuracy'] = np.mean(y_pred == y_true)
                
            except Exception as e:
                print(f"Warning: Could not compute metrics for {label}: {e}")
    
    return metrics

def main():
    print("📊 HRM-DLP Benchmark Test")
    print("=" * 60)
    
    # Initialize tester
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    test_dataset_path = "data/hrm_dlp_final/test.jsonl"
    
    try:
        tester = HRMDLPTester(checkpoint_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test dataset
    print(f"\n📁 Loading test dataset: {test_dataset_path}")
    
    if not Path(test_dataset_path).exists():
        print(f"❌ Test dataset not found: {test_dataset_path}")
        print("💡 Run with a smaller sample first:")
        print("   python test_model.py --synthetic")
        return
    
    test_examples = []
    with open(test_dataset_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                test_examples.append(example)
                
                # Limit for demo (remove this for full evaluation)
                if len(test_examples) >= 50:
                    print(f"📝 Using first 50 examples for demo")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
    
    print(f"✅ Loaded {len(test_examples)} test examples")
    
    # Run evaluation
    print(f"\n🧪 Running model evaluation...")
    
    predictions = {'sensitivity': [], 'exposure': [], 'context': [], 'obfuscation': []}
    ground_truth = {'sensitivity': [], 'exposure': [], 'context': [], 'obfuscation': []}
    
    all_decisions = []
    processing_stats = {'success': 0, 'errors': 0, 'total_act_steps': 0}
    
    for i, example in enumerate(test_examples):
        try:
            # Get model prediction
            result = tester.predict_single(example, verbose=False)
            
            # Store predictions
            for label in predictions.keys():
                predictions[label].append(result['document_scores'][label])
            
            # Store ground truth if available
            if 'labels' in example:
                for label in ground_truth.keys():
                    ground_truth[label].append(example['labels'].get(label, 0))
            
            all_decisions.append(result['decision_summary']['decision'])
            processing_stats['total_act_steps'] += result['act_steps']
            processing_stats['success'] += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(test_examples)} examples...")
                
        except Exception as e:
            print(f"❌ Error processing example {i+1}: {e}")
            processing_stats['errors'] += 1
            continue
    
    # Compute metrics
    print(f"\n📊 Computing evaluation metrics...")
    
    if processing_stats['success'] > 0:
        avg_act_steps = processing_stats['total_act_steps'] / processing_stats['success']
        print(f"✅ Successfully processed: {processing_stats['success']}/{len(test_examples)} examples")
        print(f"⚡ Average ACT steps: {avg_act_steps:.2f}")
        
        # Decision distribution
        from collections import Counter
        decision_counts = Counter(all_decisions)
        print(f"\n⚖️  Decision Distribution:")
        for decision, count in decision_counts.most_common():
            pct = (count / len(all_decisions)) * 100
            print(f"   {decision:<20}: {count:>3} ({pct:5.1f}%)")
        
        # Model performance metrics
        if any(len(ground_truth[label]) > 0 for label in ground_truth):
            print(f"\n📈 Model Performance Metrics:")
            metrics = compute_metrics(predictions, ground_truth)
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric_name:<25}: {value:.3f}")
        else:
            print(f"\n⚠️  No ground truth labels found in dataset")
            print(f"   Cannot compute precision/recall metrics")
        
        # Score distribution analysis
        print(f"\n📊 Score Distribution Analysis:")
        for label in predictions.keys():
            scores = np.array(predictions[label])
            print(f"   {label.capitalize():<12}: mean={scores.mean():.3f}, std={scores.std():.3f}, max={scores.max():.3f}")
        
        # Risk analysis
        high_risk_count = sum(1 for d in all_decisions if d == 'BLOCK')
        medium_risk_count = sum(1 for d in all_decisions if d == 'WARN')
        
        print(f"\n🎯 Risk Analysis:")
        print(f"   High Risk (BLOCK):     {high_risk_count:>3} ({high_risk_count/len(all_decisions)*100:5.1f}%)")
        print(f"   Medium Risk (WARN):    {medium_risk_count:>3} ({medium_risk_count/len(all_decisions)*100:5.1f}%)")
        
        print(f"\n✅ Benchmark completed successfully!")
        
    else:
        print(f"❌ No examples processed successfully")

if __name__ == "__main__":
    main()