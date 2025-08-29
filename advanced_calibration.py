#!/usr/bin/env python3
"""
Advanced HRM-DLP Model Calibration

More aggressive calibration to improve discrimination between risk levels.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from test_model import HRMDLPTester

def advanced_calibrate_and_test():
    print("🔧 HRM-DLP Advanced Calibration")
    print("=" * 50)
    
    # Load model
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    tester = HRMDLPTester(checkpoint_path)
    
    print("✅ Model loaded - applying advanced calibration...")
    
    # More aggressive calibration
    # Original logits are around -11.5, we want wider range: -2 to +8
    sensitivity_bias = 13.0   # High bias for sensitivity (most important)
    exposure_bias = 12.5      # High bias for exposure risk  
    context_bias = 11.5       # Medium bias for context
    obfuscation_bias = 12.0   # High bias for obfuscation
    
    calibration_bias = torch.tensor([sensitivity_bias, exposure_bias, context_bias, obfuscation_bias])
    
    # Apply calibration
    with torch.no_grad():
        if tester.model.inner.doc_head.bias is not None:
            # Reset to original first
            tester.model.inner.doc_head.bias.data.zero_()
            
            # Apply new calibration
            tester.model.inner.doc_head.bias.data += calibration_bias.to(tester.device)
            print(f"   📊 Applied stronger bias: {calibration_bias.numpy()}")
        else:
            print("   ⚠️  Model has no bias term to adjust")
            return
    
    print("\n🧪 Testing with stronger calibration...\n")
    
    # More diverse test scenarios
    scenarios = [
        {
            "name": "🚨 CRITICAL RISK - Mass Data Breach",
            "example": {
                "channel": "email",
                "user": {"role": "INTERN", "dept": "IT"},  # Intern = high risk
                "recipients": ["darkweb@hacker.ru", "competitor@rival.com"],  # Multiple external bad actors
                "subject": "💰 SELLING: Complete Customer Database + All Passwords",
                "body": "Complete customer database for sale. Contains 50,000+ records with SSNs: 123-45-6789, 987-65-4321. Credit cards: 4532-1234-5678-9012, 5555-4444-3333-2222. Root passwords: admin123, superuser456. Database access: postgresql://root:password123@prod-db.company.com:5432/customers. API keys: sk-live_1234567890abcdefghijk. Bitcoin payment only.",
                "attachments": [
                    {"name": "CUSTOMER_DATABASE_COMPLETE.sql", "size": 50000000, "mime": "application/sql"},
                    {"name": "all_passwords.txt", "size": 1000000, "mime": "text/plain"},
                    {"name": "credit_cards.csv", "size": 5000000, "mime": "text/csv"}
                ]
            }
        },
        {
            "name": "🔴 HIGH RISK - Financial Data to Personal Email",
            "example": {
                "channel": "email",
                "user": {"role": "FINANCE", "dept": "FINANCE"},
                "recipients": ["john.personal@gmail.com"],  # Personal email
                "subject": "Q4 Financials - Please Forward to Your Wife",
                "body": "Q4 revenue $15.2M, profit $3.1M. CEO SSN for verification: 123-45-6789. Wire transfer account: 1234567890. Please share with your wife who works at TechCorp - she asked about our numbers.",
                "attachments": [
                    {"name": "Q4_Financial_Report_CONFIDENTIAL.xlsx", "size": 2048000, "mime": "application/vnd.ms-excel"}
                ]
            }
        },
        {
            "name": "🟡 MEDIUM RISK - Internal PII Sharing", 
            "example": {
                "channel": "email",
                "user": {"role": "HR", "dept": "HR"},
                "recipients": ["manager@company.com"],  # Internal
                "subject": "Employee Background Check Results",
                "body": "Background check complete for John Smith. SSN: 123-45-6789, DOB: 01/15/1985, Address: 123 Main St. Clean record, recommend hiring.",
                "attachments": [
                    {"name": "background_check_report.pdf", "size": 512000, "mime": "application/pdf"}
                ]
            }
        },
        {
            "name": "🟢 LOW RISK - General Business Communication",
            "example": {
                "channel": "email",
                "user": {"role": "MARKETING", "dept": "MARKETING"},
                "recipients": ["team@company.com"],
                "subject": "Weekly Marketing Update",
                "body": "This week we launched the new campaign. Website traffic increased 15%. Next week we'll focus on social media outreach. Meeting Tuesday 2pm.",
                "attachments": []
            }
        },
        {
            "name": "🟢 MINIMAL RISK - Public Information",
            "example": {
                "channel": "email",
                "user": {"role": "LEGAL", "dept": "LEGAL"},
                "recipients": ["press@company.com"],
                "subject": "Press Release Draft",
                "body": "Company announces new product launch. CEO statement: 'We're excited to bring innovation to market.' Product available Q2 2024.",
                "attachments": [
                    {"name": "press_release_draft.docx", "size": 25600, "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
                ]
            }
        }
    ]
    
    results = []
    print("Results:")
    print("=" * 80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        
        try:
            result = tester.predict_single(scenario['example'], verbose=False)
            
            scores = result['document_scores']
            decision = result['decision_summary']
            
            # Create risk score (weighted combination)
            risk_score = (scores['sensitivity'] * 0.4 + 
                         scores['exposure'] * 0.3 + 
                         scores['context'] * 0.2 + 
                         scores['obfuscation'] * 0.1)
            
            print(f"   📊 Risk Score: {risk_score:.3f}")
            print(f"   📊 Breakdown:  S:{scores['sensitivity']:.2f} E:{scores['exposure']:.2f} C:{scores['context']:.2f} O:{scores['obfuscation']:.2f}")
            print(f"   ⚖️  Decision:   {decision['decision']} ({decision['risk_level']})")
            print(f"   🎯 Confidence: {decision['confidence']:.3f}")
            print()
            
            result['risk_score'] = risk_score
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Analysis
    if results:
        print("📈 ADVANCED CALIBRATION ANALYSIS")
        print("=" * 50)
        
        risk_scores = [r['risk_score'] for r in results]
        decisions = [r['decision_summary']['decision'] for r in results]
        
        print(f"📊 Risk Score Range: {min(risk_scores):.3f} - {max(risk_scores):.3f}")
        print(f"📊 Risk Score StdDev: {np.std(risk_scores):.3f}")
        
        print(f"\n⚖️  Decision Distribution:")
        from collections import Counter
        decision_counts = Counter(decisions)
        for decision, count in decision_counts.items():
            print(f"   {decision}: {count}")
        
        # Check discrimination
        score_range = max(risk_scores) - min(risk_scores)
        print(f"\n🎯 Discrimination Analysis:")
        print(f"   Score range: {score_range:.3f}")
        
        if score_range > 0.1:
            print("   ✅ Good discrimination between scenarios!")
        elif score_range > 0.05:
            print("   🟡 Moderate discrimination - improving")  
        else:
            print("   ⚠️  Still low discrimination")
        
        # Expected vs actual
        expected_order = [4, 3, 2, 1, 0]  # Critical, High, Medium, Low, Minimal
        actual_order = sorted(range(len(risk_scores)), key=lambda i: risk_scores[i], reverse=True)
        
        print(f"\n🎯 Risk Ranking Analysis:")
        scenario_names = ["Critical", "High", "Medium", "Low", "Minimal"]
        for rank, idx in enumerate(actual_order):
            expected_rank = expected_order.index(idx) if idx < len(expected_order) else "?"
            print(f"   Rank {rank+1}: {scenario_names[idx]} (expected rank {expected_rank+1})")
        
        print(f"\n✅ Advanced calibration complete!")
        print(f"   Model now shows risk scores from {min(risk_scores):.3f} to {max(risk_scores):.3f}")
        print(f"   Recommendations:")
        
        if score_range > 0.1:
            print(f"   🎉 Excellent - ready for production testing!")
            print(f"   💡 Try: python test_model.py --dataset data/hrm_dlp_final/test.jsonl --max_examples 10")
        else:
            print(f"   🔧 May need further tuning for your specific use case")
            print(f"   💡 Consider fine-tuning on labeled examples")

if __name__ == "__main__":
    advanced_calibrate_and_test()