#!/usr/bin/env python3
"""
Final Calibrated Model Test

Tests the temperature-calibrated HRM-DLP model on comprehensive scenarios.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from test_model import HRMDLPTester

# Optimized calibration parameters from temperature scaling
CALIBRATED_TEMPERATURES = np.array([0.26824779, 5.0, 5.0, 5.0])
CALIBRATED_BIASES = np.array([5.0, -1.68696815, 5.0, -5.0])

def apply_calibration(tester):
    """Apply temperature scaling calibration to model."""
    original_predict = tester.predict_single
    
    def calibrated_predict(example, verbose=True):
        result = original_predict(example, verbose=False)
        
        # Apply temperature scaling
        probs = np.array([
            result['document_scores']['sensitivity'],
            result['document_scores']['exposure'],
            result['document_scores']['context'], 
            result['document_scores']['obfuscation']
        ])
        
        # Convert to logits, apply scaling, convert back
        probs = np.clip(probs, 1e-7, 1-1e-7)
        logits = np.log(probs / (1 - probs))
        scaled_logits = logits / CALIBRATED_TEMPERATURES + CALIBRATED_BIASES
        calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
        
        # Update results
        result['document_scores'] = {
            'sensitivity': float(calibrated_probs[0]),
            'exposure': float(calibrated_probs[1]),
            'context': float(calibrated_probs[2]),
            'obfuscation': float(calibrated_probs[3])
        }
        
        # Recalculate decision with new scores
        result['decision_summary'] = tester._make_decision(calibrated_probs)
        
        if verbose:
            tester._print_predictions(result, "")
            
        return result
    
    tester.predict_single = calibrated_predict
    return tester

def create_comprehensive_test_scenarios():
    """Create comprehensive test scenarios across risk spectrum."""
    return [
        {
            "name": "🚨 EXTREME - Mass Data Exfiltration",
            "expected_risk": 0.95,
            "expected_decision": "BLOCK",
            "example": {
                "channel": "email",
                "user": {"role": "INTERN", "dept": "IT"},
                "recipients": ["hacker@dark.web", "competitor@rival.com", "buyer@criminal.org"],
                "subject": "💰 MASSIVE DATA SALE - Customer DB + Passwords + Financial Data",
                "body": "URGENT SALE: Complete company data package. Customer database (2M+ records): SSNs 123-45-6789, 987-65-4321, 555-44-3333, credit cards 4532-1234-5678-9012, 5555-4444-3333-2222. Employee data: passwords admin123, root456, executive salaries, trade secrets. Database: postgresql://root:SuperSecret@prod.company.com:5432/everything. APIs: sk-live_abcd1234, AWS AKIAIOSFODNN7EXAMPLE. Bitcoin preferred. $100K minimum.",
                "attachments": [
                    {"name": "customer_database_2M_records.sql", "size": 200000000, "mime": "application/sql"},
                    {"name": "employee_passwords_and_ssns.txt", "size": 5000000, "mime": "text/plain"},
                    {"name": "credit_card_numbers.csv", "size": 50000000, "mime": "text/csv"},
                    {"name": "aws_keys_production.json", "size": 1024, "mime": "application/json"}
                ]
            }
        },
        {
            "name": "🔴 HIGH - Financial Data to Personal Email", 
            "expected_risk": 0.80,
            "expected_decision": "BLOCK",
            "example": {
                "channel": "email",
                "user": {"role": "CFO", "dept": "FINANCE"},
                "recipients": ["my.personal@gmail.com"],
                "subject": "Confidential: Q4 financials and merger details",
                "body": "Q4 results for my records: Revenue $75M, profit $18M. Merger with TechCorp valued at $500M - closing next month. Executive bonuses: CEO $3M, I get $2.5M. Bank details for wire: Chase 987654321, routing 021000021. For verification, my SSN: 123-45-6789. Keep private until announcement.",
                "attachments": [
                    {"name": "Q4_CONFIDENTIAL_financials.xlsx", "size": 10240000, "mime": "application/vnd.ms-excel"}
                ]
            }
        },
        {
            "name": "🟠 MEDIUM-HIGH - Source Code + Credentials to Competitor",
            "expected_risk": 0.65, 
            "expected_decision": "WARN",
            "example": {
                "channel": "email",
                "user": {"role": "SENIOR_DEV", "dept": "ENGINEERING"},
                "recipients": ["hiring@competitor.com"],
                "subject": "Portfolio: Proprietary ML algorithms and architecture",
                "body": "Here's my portfolio for the senior engineer role. Includes our proprietary recommendation engine (40% better than Netflix), real-time analytics system, and fraud detection ML models. Database connections: mysql://analytics:AnalyticsPwd2024@internal.company.com:3306/ml_models. These algorithms took 3 years to develop and are our competitive advantage.",
                "attachments": [
                    {"name": "recommendation_engine_proprietary.py", "size": 2048000, "mime": "text/x-python"},
                    {"name": "ml_fraud_detection.py", "size": 1024000, "mime": "text/x-python"},
                    {"name": "analytics_architecture.pdf", "size": 5120000, "mime": "application/pdf"}
                ]
            }
        },
        {
            "name": "🟡 MEDIUM - Employee PII Internal Sharing",
            "expected_risk": 0.45,
            "expected_decision": "WARN",
            "example": {
                "channel": "email",
                "user": {"role": "HR", "dept": "HR"},
                "recipients": ["payroll@company.com", "benefits@company.com"],
                "subject": "Q1 salary adjustments and employee data",
                "body": "Q1 salary changes: John Smith (SSN: 123-45-6789) promoted to $125K, Jane Doe (SSN: 987-65-4321) relocated, new address 456 Oak St, phone 555-123-4567. Michael Brown under review, SSN: 555-44-3333. Please update payroll systems accordingly.",
                "attachments": [
                    {"name": "employee_salary_changes_Q1.xlsx", "size": 512000, "mime": "application/vnd.ms-excel"}
                ]
            }
        },
        {
            "name": "🟢 LOW-MEDIUM - Business Metrics to Trusted Partner",
            "expected_risk": 0.25,
            "expected_decision": "ALLOW_WITH_MONITORING",
            "example": {
                "channel": "email",
                "user": {"role": "VP_SALES", "dept": "SALES"},
                "recipients": ["partner@trusted-vendor.com"],
                "subject": "Q1 performance metrics and partnership review",
                "body": "Q1 partnership review: Revenue from your referrals $2.1M (up 15%), customer satisfaction 94%, avg deal size $45K. Team headcount now 85 sales reps. Looking forward to expanding our partnership in Q2. Conference call scheduled for next Friday 3pm EST.",
                "attachments": [
                    {"name": "partnership_metrics_Q1.pdf", "size": 256000, "mime": "application/pdf"}
                ]
            }
        },
        {
            "name": "🟢 LOW - Internal Project Update",
            "expected_risk": 0.15,
            "expected_decision": "ALLOW",
            "example": {
                "channel": "email", 
                "user": {"role": "PROJECT_MANAGER", "dept": "PRODUCT"},
                "recipients": ["dev-team@company.com", "design@company.com"],
                "subject": "Project Beta status update - on track",
                "body": "Project Beta update: Frontend 85% complete, backend API testing in progress, mobile app in design review. Target launch Q2 remains achievable. Budget tracking well at $1.2M of $1.5M allocated. Next sprint planning Monday 10am.",
                "attachments": []
            }
        },
        {
            "name": "🟢 MINIMAL - Public Information",
            "expected_risk": 0.05,
            "expected_decision": "ALLOW",
            "example": {
                "channel": "email",
                "user": {"role": "COMMUNICATIONS", "dept": "MARKETING"},
                "recipients": ["press@company.com", "media@tech-news.com"],
                "subject": "Press release: New product announcement",
                "body": "FOR IMMEDIATE RELEASE: Company announces innovative new product launching Q2 2024. CEO statement: 'We're excited to bring this innovation to market and serve our customers better.' Product demonstrations available at upcoming trade show.",
                "attachments": [
                    {"name": "press_release_final.pdf", "size": 128000, "mime": "application/pdf"}
                ]
            }
        }
    ]

def final_comprehensive_test():
    print("🎯 HRM-DLP Final Calibrated Model Test")
    print("=" * 70)
    
    # Load and calibrate model
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    tester = HRMDLPTester(checkpoint_path)
    tester = apply_calibration(tester)
    
    print("✅ Model loaded and calibrated successfully!")
    print(f"🔧 Applied temperature scaling: {CALIBRATED_TEMPERATURES}")
    print(f"🔧 Applied bias adjustments: {CALIBRATED_BIASES}")
    
    # Test scenarios
    scenarios = create_comprehensive_test_scenarios()
    print(f"\n📋 Testing {len(scenarios)} comprehensive scenarios...")
    print("=" * 70)
    
    results = []
    correct_decisions = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Expected Risk: {scenario['expected_risk']:.3f}")
        print(f"   Expected Decision: {scenario['expected_decision']}")
        
        try:
            result = tester.predict_single(scenario['example'], verbose=False)
            
            # Calculate overall risk score
            scores = result['document_scores']
            risk_score = (scores['sensitivity'] * 0.4 +
                         scores['exposure'] * 0.3 + 
                         scores['context'] * 0.2 +
                         scores['obfuscation'] * 0.1)
            
            decision = result['decision_summary']['decision']
            risk_level = result['decision_summary']['risk_level']
            
            # Check if decision matches expectation
            decision_correct = decision == scenario['expected_decision']
            if decision_correct:
                correct_decisions += 1
            
            print(f"   📊 Predicted Risk: {risk_score:.3f}")
            print(f"   📊 Breakdown: S:{scores['sensitivity']:.3f} E:{scores['exposure']:.3f} C:{scores['context']:.3f} O:{scores['obfuscation']:.3f}")
            print(f"   ⚖️  Actual Decision: {decision} ({risk_level})")
            print(f"   🎯 Risk Error: {abs(risk_score - scenario['expected_risk']):.3f}")
            print(f"   {'✅' if decision_correct else '❌'} Decision {'Correct' if decision_correct else 'Incorrect'}")
            
            result['risk_score'] = risk_score
            result['expected_risk'] = scenario['expected_risk'] 
            result['expected_decision'] = scenario['expected_decision']
            result['decision_correct'] = decision_correct
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Final analysis
    if results:
        print(f"\n📈 FINAL CALIBRATED MODEL ANALYSIS")
        print("=" * 70)
        
        risk_scores = [r['risk_score'] for r in results]
        expected_risks = [r['expected_risk'] for r in results]
        errors = [abs(r['risk_score'] - r['expected_risk']) for r in results]
        
        # Performance metrics
        print(f"🎯 Performance Metrics:")
        print(f"   Risk Score Range:        {min(risk_scores):.3f} - {max(risk_scores):.3f}")
        print(f"   Risk Score Std Dev:      {np.std(risk_scores):.3f}")
        print(f"   Mean Absolute Error:     {np.mean(errors):.3f}")
        print(f"   Decision Accuracy:       {correct_decisions}/{len(results)} ({correct_decisions/len(results)*100:.1f}%)")
        print(f"   Risk Correlation:        {np.corrcoef(risk_scores, expected_risks)[0,1]:.3f}")
        
        # Risk distribution analysis
        print(f"\n📊 Risk Distribution:")
        from collections import Counter
        decisions = [r['decision_summary']['decision'] for r in results]
        decision_counts = Counter(decisions)
        for decision, count in decision_counts.items():
            pct = (count / len(results)) * 100
            print(f"   {decision:<20}: {count:>2} ({pct:5.1f}%)")
        
        # Discrimination analysis
        score_range = max(risk_scores) - min(risk_scores)
        print(f"\n🎯 Model Discrimination Analysis:")
        if score_range > 0.4:
            status = "🎉 EXCELLENT"
            recommendation = "Ready for production deployment!"
        elif score_range > 0.3:
            status = "🟢 GOOD"  
            recommendation = "Strong discrimination, suitable for production"
        elif score_range > 0.2:
            status = "🟡 MODERATE"
            recommendation = "Acceptable discrimination, monitor in production"
        else:
            status = "🟠 LIMITED"
            recommendation = "Consider additional calibration or training"
        
        print(f"   Score Range: {score_range:.3f} - {status}")
        print(f"   💡 {recommendation}")
        
        # Error analysis by risk level
        print(f"\n📉 Error Analysis by Risk Level:")
        high_risk_errors = [abs(r['risk_score'] - r['expected_risk']) for r in results if r['expected_risk'] > 0.7]
        medium_risk_errors = [abs(r['risk_score'] - r['expected_risk']) for r in results if 0.3 <= r['expected_risk'] <= 0.7]
        low_risk_errors = [abs(r['risk_score'] - r['expected_risk']) for r in results if r['expected_risk'] < 0.3]
        
        if high_risk_errors:
            print(f"   High Risk (>0.7):   MAE = {np.mean(high_risk_errors):.3f}")
        if medium_risk_errors:
            print(f"   Medium Risk (0.3-0.7): MAE = {np.mean(medium_risk_errors):.3f}")
        if low_risk_errors:
            print(f"   Low Risk (<0.3):    MAE = {np.mean(low_risk_errors):.3f}")
        
        print(f"\n✅ Final calibrated model testing completed!")
        print(f"💾 Save these calibration parameters for production:")
        print(f"   TEMPERATURES = {CALIBRATED_TEMPERATURES}")
        print(f"   BIASES = {CALIBRATED_BIASES}")
        
    else:
        print("❌ No successful predictions made")

if __name__ == "__main__":
    final_comprehensive_test()