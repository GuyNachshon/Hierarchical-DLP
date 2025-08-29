#!/usr/bin/env python3
"""
HRM-DLP Model Calibration

Adjusts model output biases to provide more realistic risk scores.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from test_model import HRMDLPTester

def calibrate_and_test():
    print("🔧 HRM-DLP Model Calibration")
    print("=" * 50)
    
    # Load model
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    tester = HRMDLPTester(checkpoint_path)
    
    print("✅ Model loaded - applying calibration...")
    
    # Calibration: Add bias to document head to shift predictions
    # Current logits are ~-11.5, we want them around 0 to +5 range
    calibration_bias = torch.tensor([11.0, 10.5, 10.0, 10.5])  # Shift each score type differently
    
    # Apply calibration by modifying the doc_head bias
    with torch.no_grad():
        if tester.model.inner.doc_head.bias is not None:
            original_bias = tester.model.inner.doc_head.bias.clone()
            tester.model.inner.doc_head.bias.data += calibration_bias.to(tester.device)
            print(f"   📊 Applied bias adjustment: {calibration_bias.numpy()}")
        else:
            print("   ⚠️  Model has no bias term to adjust")
            return
    
    print("\n🧪 Testing calibrated model on risk scenarios...\n")
    
    # Test scenarios
    scenarios = [
        {
            "name": "🔴 EXTREME HIGH RISK",
            "example": {
                "channel": "email",
                "user": {"role": "FINANCE", "dept": "FINANCE"},
                "recipients": ["hacker@evil.com"],
                "subject": "All Passwords and Customer Data",
                "body": "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012, Password: admin123, Database: postgres://admin:secret@db.company.com/sensitive",
                "attachments": [{"name": "customer_data.xlsx", "size": 5000000, "mime": "application/vnd.ms-excel"}]
            }
        },
        {
            "name": "🟡 MEDIUM RISK",
            "example": {
                "channel": "email",
                "user": {"role": "HR", "dept": "HR"},
                "recipients": ["manager@company.com"],
                "subject": "Employee Information",
                "body": "Employee ID: EMP-12345, Phone: 555-123-4567. Performance review scheduled for next week.",
                "attachments": []
            }
        },
        {
            "name": "🟢 LOW RISK",
            "example": {
                "channel": "email",
                "user": {"role": "MARKETING", "dept": "MARKETING"},
                "recipients": ["team@company.com"],
                "subject": "Weekly Update",
                "body": "Team meeting scheduled for Tuesday. Please review the project timeline and provide updates.",
                "attachments": []
            }
        }
    ]
    
    results = []
    for scenario in scenarios:
        print(f"{scenario['name']}")
        print("-" * 40)
        
        try:
            result = tester.predict_single(scenario['example'], verbose=False)
            
            scores = result['document_scores']
            decision = result['decision_summary']
            
            print(f"   📊 Sensitivity: {scores['sensitivity']:.3f}")
            print(f"   📊 Exposure:    {scores['exposure']:.3f}")
            print(f"   📊 Context:     {scores['context']:.3f}")
            print(f"   📊 Obfuscation: {scores['obfuscation']:.3f}")
            print(f"   ⚖️  Decision:    {decision['decision']} ({decision['risk_level']})")
            print(f"   🎯 Confidence:  {decision['confidence']:.3f}")
            print(f"   ⚡ ACT Steps:   {result['act_steps']}")
            print()
            
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Summary
    if results:
        print("📈 CALIBRATION SUMMARY")
        print("=" * 40)
        
        decisions = [r['decision_summary']['decision'] for r in results]
        avg_scores = {}
        for score_type in ['sensitivity', 'exposure', 'context', 'obfuscation']:
            avg_scores[score_type] = np.mean([r['document_scores'][score_type] for r in results])
        
        print(f"✅ Processed {len(results)} scenarios successfully")
        print(f"📊 Average scores:")
        for score_type, avg in avg_scores.items():
            print(f"   {score_type.capitalize():<12}: {avg:.3f}")
        
        print(f"⚖️  Decisions: {decisions}")
        print(f"⚡ Average ACT steps: {np.mean([r['act_steps'] for r in results]):.1f}")
        
        # Check if we have good discrimination
        sensitivity_range = max([r['document_scores']['sensitivity'] for r in results]) - min([r['document_scores']['sensitivity'] for r in results])
        exposure_range = max([r['document_scores']['exposure'] for r in results]) - min([r['document_scores']['exposure'] for r in results])
        
        print(f"\n🎯 Discrimination Analysis:")
        print(f"   Sensitivity range: {sensitivity_range:.3f}")
        print(f"   Exposure range:    {exposure_range:.3f}")
        
        if sensitivity_range > 0.2 or exposure_range > 0.2:
            print("   ✅ Good score discrimination - calibration working!")
        else:
            print("   ⚠️  Low discrimination - may need stronger calibration")
    
    print("\n💡 Next Steps:")
    print("   1. If results look good, save the calibrated model")
    print("   2. Test on more examples with: python test_model.py --synthetic")
    print("   3. Run full benchmark: python benchmark_test.py")

if __name__ == "__main__":
    calibrate_and_test()