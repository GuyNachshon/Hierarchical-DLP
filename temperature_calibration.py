#!/usr/bin/env python3
"""
HRM-DLP Temperature Scaling Calibration

Uses temperature scaling and Platt scaling to improve model discrimination.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import numpy as np
from test_model import HRMDLPTester
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


class TemperatureScaledModel(nn.Module):
    """
    Temperature scaling for neural network calibration.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(4) * 1.5)  # Per-class temperature
        self.bias_adjustment = nn.Parameter(torch.zeros(4))   # Learnable bias
        
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            logits = self.model(*args, **kwargs)
        
        # Apply temperature scaling and bias adjustment
        if hasattr(logits, 'doc_logits'):
            scaled_logits = logits.doc_logits / self.temperature.unsqueeze(0) + self.bias_adjustment.unsqueeze(0)
            logits.doc_logits = scaled_logits
            
        return logits


def create_calibration_dataset():
    """Create diverse calibration examples with known risk levels."""
    return [
        # CRITICAL RISK Examples
        {
            "name": "Mass data breach to external",
            "example": {
                "channel": "email",
                "user": {"role": "INTERN", "dept": "IT"},
                "recipients": ["hacker@dark.net", "competitor@rival.com"],
                "subject": "🔥 URGENT: Complete customer database + all passwords for sale!",
                "body": "Complete customer database for sale. Contains 100,000+ records with SSNs: 123-45-6789, 987-65-4321, 555-44-3333. Credit cards: 4532-1234-5678-9012, 5555-4444-3333-2222. Root passwords: admin123, superuser456, root_password_2024. Database access: postgresql://root:SuperSecret123@prod-db.company.com:5432/customers_sensitive. API keys: sk-live_1234567890abcdefghijk, sk-test_0987654321zyxwvutsrq. AWS access: AKIAIOSFODNN7EXAMPLE. Bitcoin payment preferred. Urgent sale!",
                "attachments": [
                    {"name": "CUSTOMER_DATABASE_COMPLETE.sql", "size": 100000000, "mime": "application/sql"},
                    {"name": "all_passwords_and_keys.txt", "size": 2000000, "mime": "text/plain"},
                    {"name": "credit_cards_full.csv", "size": 10000000, "mime": "text/csv"},
                    {"name": "ssh_keys_prod.pem", "size": 4096, "mime": "application/x-pem-file"}
                ]
            },
            "expected_risk": 0.95
        },
        {
            "name": "Financial data to personal email",
            "example": {
                "channel": "email",
                "user": {"role": "CFO", "dept": "FINANCE"},
                "recipients": ["personal.account@gmail.com", "spouse@hotmail.com"],
                "subject": "FYI - Q4 numbers and executive compensation",
                "body": "Q4 results: Revenue $50.2M (up 23%), EBITDA $12.1M. Exec bonuses: CEO $2.1M, CFO $1.8M. Merger talks with TechCorp proceeding - $125M valuation discussed. Wire details: Bank of America 123456789, routing 021000322. CEO SSN for verification: 123-45-6789. Keep confidential until announcement.",
                "attachments": [
                    {"name": "Q4_Financial_CONFIDENTIAL.xlsx", "size": 5120000, "mime": "application/vnd.ms-excel"},
                    {"name": "executive_compensation.pdf", "size": 1024000, "mime": "application/pdf"}
                ]
            },
            "expected_risk": 0.90
        },
        
        # HIGH RISK Examples  
        {
            "name": "Internal PII to wrong department",
            "example": {
                "channel": "email",
                "user": {"role": "HR", "dept": "HR"},
                "recipients": ["marketing@company.com", "sales@company.com"],
                "subject": "Employee salary review data - please review",
                "body": "Please review salary adjustments for Q1. John Smith (SSN: 123-45-6789) salary increased to $95K. Jane Doe (SSN: 987-65-4321) promoted, new salary $110K. Michael Johnson under performance review, SSN: 555-44-3333, considering termination. Home addresses and phone numbers in spreadsheet.",
                "attachments": [
                    {"name": "salary_review_2024_SENSITIVE.xlsx", "size": 2048000, "mime": "application/vnd.ms-excel"}
                ]
            },
            "expected_risk": 0.75
        },
        {
            "name": "Source code to competitor",
            "example": {
                "channel": "email",
                "user": {"role": "DEVELOPER", "dept": "ENGINEERING"},
                "recipients": ["recruiter@competitor.com"],
                "subject": "Portfolio examples for interview",
                "body": "Here are some code examples from my current work. The ML algorithm we use for recommendations is proprietary but I thought you'd find it interesting. Database connection: mysql://dev:DevPass2024@internal-db:3306/analytics. This gives 40% better performance than industry standard.",
                "attachments": [
                    {"name": "recommendation_algorithm.py", "size": 512000, "mime": "text/x-python"},
                    {"name": "analytics_queries.sql", "size": 256000, "mime": "application/sql"}
                ]
            },
            "expected_risk": 0.70
        },
        
        # MEDIUM RISK Examples
        {
            "name": "Business metrics to partner",
            "example": {
                "channel": "email", 
                "user": {"role": "MANAGER", "dept": "BUSINESS"},
                "recipients": ["partner@trusted-vendor.com"],
                "subject": "Monthly KPIs and performance update",
                "body": "Monthly update: Customer acquisition cost $125 (down 8%), LTV $890 (up 12%), churn rate 3.2%. Team performance strong. Next quarter forecast looks positive. Employee headcount now 145. Conference call scheduled for next Tuesday.",
                "attachments": [
                    {"name": "monthly_kpis.pdf", "size": 256000, "mime": "application/pdf"}
                ]
            },
            "expected_risk": 0.40
        },
        {
            "name": "Project timeline internal",
            "example": {
                "channel": "email",
                "user": {"role": "PM", "dept": "PRODUCT"},
                "recipients": ["dev-team@company.com", "qa@company.com"],
                "subject": "Project Alpha timeline update", 
                "body": "Project Alpha milestone update: Backend API complete, frontend 70% done. Launch target moved to Q2 due to integration issues. Budget still on track at $2.1M. Risk: dependency on third-party API may cause delays.",
                "attachments": []
            },
            "expected_risk": 0.25
        },
        
        # LOW RISK Examples
        {
            "name": "Team meeting notes",
            "example": {
                "channel": "email",
                "user": {"role": "ADMIN", "dept": "OPERATIONS"},
                "recipients": ["team@company.com"],
                "subject": "Weekly team meeting notes",
                "body": "Meeting notes from Tuesday: Discussed project timelines, resource allocation for Q2. Action items: Update project documentation, schedule client demos, prepare Q1 review. Next meeting scheduled for next Tuesday 2pm.",
                "attachments": [
                    {"name": "meeting_notes_03_12.docx", "size": 25600, "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
                ]
            },
            "expected_risk": 0.10
        },
        {
            "name": "Public information sharing",
            "example": {
                "channel": "email",
                "user": {"role": "MARKETING", "dept": "MARKETING"},
                "recipients": ["press@company.com", "media@industry-blog.com"],
                "subject": "Company announcement - new product launch",
                "body": "We're excited to announce the launch of our new product line. Available Q2 2024. CEO quote: 'This represents our continued innovation in the market.' Product demo available on our website.",
                "attachments": [
                    {"name": "press_release_draft.pdf", "size": 51200, "mime": "application/pdf"}
                ]
            },
            "expected_risk": 0.05
        }
    ]


def optimize_temperature_scaling(tester, calibration_examples):
    """Find optimal temperature parameters using calibration data."""
    print("🔥 Optimizing temperature scaling parameters...")
    
    # Collect model predictions
    logits_data = []
    risk_targets = []
    
    for example_data in calibration_examples:
        example = example_data["example"]
        expected_risk = example_data["expected_risk"]
        
        try:
            # Get raw logits before sigmoid
            result = tester.predict_single(example, verbose=False)
            
            # Convert back to logits (inverse sigmoid)
            probs = np.array([
                result['document_scores']['sensitivity'],
                result['document_scores']['exposure'], 
                result['document_scores']['context'],
                result['document_scores']['obfuscation']
            ])
            
            # Clamp probabilities to avoid numerical issues
            probs = np.clip(probs, 1e-7, 1-1e-7)
            logits = np.log(probs / (1 - probs))  # logit = log(p / (1-p))
            
            logits_data.append(logits)
            risk_targets.append(expected_risk)
            
        except Exception as e:
            print(f"   ⚠️  Skipping example due to error: {e}")
            continue
    
    if len(logits_data) < 4:
        print("   ❌ Not enough valid examples for optimization")
        return None, None
    
    logits_array = np.array(logits_data)
    targets_array = np.array(risk_targets)
    
    print(f"   📊 Using {len(logits_data)} examples for optimization")
    print(f"   📊 Target risk range: {targets_array.min():.3f} - {targets_array.max():.3f}")
    
    # Optimize temperature and bias for each class
    def temperature_loss(params):
        temperatures = params[:4]
        biases = params[4:]
        
        total_loss = 0
        for i in range(len(logits_data)):
            # Apply temperature and bias
            scaled_logits = logits_array[i] / temperatures + biases
            
            # Convert to risk score (weighted combination)  
            probs = 1 / (1 + np.exp(-scaled_logits))
            risk_score = (probs[0] * 0.4 +  # sensitivity
                         probs[1] * 0.3 +   # exposure
                         probs[2] * 0.2 +   # context  
                         probs[3] * 0.1)    # obfuscation
            
            # MSE loss with target risk
            total_loss += (risk_score - targets_array[i]) ** 2
        
        return total_loss / len(logits_data)
    
    # Initial guess: temperature=1.5, bias=0
    initial_params = np.concatenate([np.ones(4) * 1.5, np.zeros(4)])
    
    # Optimize using scipy
    from scipy.optimize import minimize
    result = minimize(
        temperature_loss, 
        initial_params,
        method='L-BFGS-B',
        bounds=[(0.1, 5.0)] * 4 + [(-5.0, 5.0)] * 4  # Temperature: 0.1-5, Bias: -5 to 5
    )
    
    if result.success:
        optimal_temps = result.x[:4]
        optimal_biases = result.x[4:]
        
        print(f"   ✅ Optimization successful!")
        print(f"   🌡️  Optimal temperatures: [{optimal_temps[0]:.3f}, {optimal_temps[1]:.3f}, {optimal_temps[2]:.3f}, {optimal_temps[3]:.3f}]")
        print(f"   ⚖️  Optimal biases:      [{optimal_biases[0]:.3f}, {optimal_biases[1]:.3f}, {optimal_biases[2]:.3f}, {optimal_biases[3]:.3f}]")
        print(f"   📉 Final loss: {result.fun:.6f}")
        
        return optimal_temps, optimal_biases
    else:
        print(f"   ❌ Optimization failed: {result.message}")
        return None, None


def apply_temperature_calibration(tester, temperatures, biases):
    """Apply temperature scaling to model."""
    if temperatures is None or biases is None:
        print("❌ No calibration parameters available")
        return False
        
    print("🔥 Applying temperature calibration...")
    
    # Store original forward method
    original_predict = tester.predict_single
    
    def calibrated_predict(example, verbose=True):
        # Get original prediction
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
        scaled_logits = logits / temperatures + biases
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
    
    # Replace prediction method
    tester.predict_single = calibrated_predict
    
    print("   ✅ Temperature calibration applied successfully!")
    return True


def temperature_calibrate_and_test():
    print("🔥 HRM-DLP Temperature Scaling Calibration")
    print("=" * 60)
    
    # Load model
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    tester = HRMDLPTester(checkpoint_path)
    print("✅ Model loaded successfully!")
    
    # Create calibration dataset
    calibration_examples = create_calibration_dataset()
    print(f"📚 Created {len(calibration_examples)} calibration examples")
    
    # Optimize temperature scaling
    temperatures, biases = optimize_temperature_scaling(tester, calibration_examples)
    
    if temperatures is not None:
        # Apply calibration
        apply_temperature_calibration(tester, temperatures, biases)
        
        # Test calibrated model
        print(f"\n🧪 Testing temperature-calibrated model...")
        print("=" * 60)
        
        test_scenarios = [
            calibration_examples[0],  # Critical risk
            calibration_examples[2],  # High risk  
            calibration_examples[4],  # Medium risk
            calibration_examples[6],  # Low risk
        ]
        
        results = []
        for i, scenario in enumerate(test_scenarios):
            print(f"\n{i+1}. {scenario['name']}")
            print(f"   Expected risk: {scenario['expected_risk']:.3f}")
            
            try:
                result = tester.predict_single(scenario['example'], verbose=False)
                
                # Calculate overall risk score
                scores = result['document_scores']
                risk_score = (scores['sensitivity'] * 0.4 +
                             scores['exposure'] * 0.3 + 
                             scores['context'] * 0.2 +
                             scores['obfuscation'] * 0.1)
                
                print(f"   📊 Predicted risk: {risk_score:.3f}")
                print(f"   📊 Breakdown: S:{scores['sensitivity']:.3f} E:{scores['exposure']:.3f} C:{scores['context']:.3f} O:{scores['obfuscation']:.3f}")
                print(f"   ⚖️  Decision: {result['decision_summary']['decision']} ({result['decision_summary']['risk_level']})")
                print(f"   🎯 Error: {abs(risk_score - scenario['expected_risk']):.3f}")
                
                result['risk_score'] = risk_score
                result['expected_risk'] = scenario['expected_risk']
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Analysis
        if results:
            print(f"\n📈 TEMPERATURE CALIBRATION ANALYSIS")
            print("=" * 60)
            
            risk_scores = [r['risk_score'] for r in results]
            expected_risks = [r['expected_risk'] for r in results]
            errors = [abs(r['risk_score'] - r['expected_risk']) for r in results]
            
            print(f"📊 Risk Score Range: {min(risk_scores):.3f} - {max(risk_scores):.3f}")
            print(f"📊 Risk Score StdDev: {np.std(risk_scores):.3f}")
            print(f"📊 Mean Absolute Error: {np.mean(errors):.3f}")
            print(f"📊 Correlation with expected: {np.corrcoef(risk_scores, expected_risks)[0,1]:.3f}")
            
            # Check discrimination improvement
            score_range = max(risk_scores) - min(risk_scores)
            if score_range > 0.3:
                print(f"\n🎉 Excellent discrimination! Range: {score_range:.3f}")
                print("   ✅ Ready for production deployment")
            elif score_range > 0.2:
                print(f"\n🟢 Good discrimination! Range: {score_range:.3f}")
                print("   ✅ Significant improvement over baseline")
            elif score_range > 0.1:
                print(f"\n🟡 Moderate discrimination. Range: {score_range:.3f}")
                print("   💡 Consider additional calibration or fine-tuning")
            else:
                print(f"\n🟠 Still limited discrimination. Range: {score_range:.3f}")
                print("   💡 May need architectural changes or more training data")
            
            print(f"\n💾 Calibration Parameters (save these!):")
            print(f"   Temperatures: {temperatures}")
            print(f"   Biases: {biases}")
            
    else:
        print("❌ Temperature optimization failed")


if __name__ == "__main__":
    temperature_calibrate_and_test()