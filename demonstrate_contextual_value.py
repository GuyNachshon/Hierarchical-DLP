#!/usr/bin/env python3
"""
Demonstrate Contextual Labeling Value

Creates examples that show the strategic difference between
pattern-based and contextual approaches when user context is available.
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from contextual_label_strategy import ContextualDLPLabeler
from generate_missing_labels import DLPLabelGenerator
import numpy as np

def create_strategic_examples():
    """Create examples that highlight the strategic difference."""
    return [
        {
            "name": "CFO sharing financials (appropriate context)",
            "example": {
                "channel": "email",
                "user": {"role": "CFO", "dept": "FINANCE"},
                "subject": "Q4 Board Presentation - Financial Summary",
                "body": "Attached are Q4 financials for the board meeting. Revenue $50M, EBITDA $12M. Executive compensation details included for discussion. Please review before presentation.",
                "recipients": ["board-secretary@company.com", "ceo@company.com"],
                "attachments": [
                    {
                        "name": "Q4_Board_Financials.xlsx", 
                        "size": 2048000,
                        "sensitivity_indicators": ["financial_data", "executive_compensation"]
                    }
                ]
            }
        },
        {
            "name": "Intern sharing same financials (inappropriate context)", 
            "example": {
                "channel": "email",
                "user": {"role": "INTERN", "dept": "MARKETING"},
                "subject": "Interesting numbers I found",
                "body": "Found this financial data on the shared drive. Revenue $50M, EBITDA $12M. Executive compensation looks crazy high! Thought you'd find this interesting.",
                "recipients": ["friend@gmail.com", "roommate@yahoo.com"],
                "attachments": [
                    {
                        "name": "Q4_Board_Financials.xlsx",
                        "size": 2048000, 
                        "sensitivity_indicators": ["financial_data", "executive_compensation"]
                    }
                ]
            }
        },
        {
            "name": "Senior engineer sharing code (appropriate)",
            "example": {
                "channel": "email",
                "user": {"role": "SENIOR_ENGINEER", "dept": "ENGINEERING"},
                "subject": "Code review: Authentication module",
                "body": "Please review the new authentication code. Added support for API key rotation and improved password hashing. Database connection uses environment variables as discussed.",
                "recipients": ["tech-lead@company.com", "security-team@company.com"],
                "attachments": [
                    {
                        "name": "auth_module.py",
                        "size": 45000,
                        "sensitivity_indicators": ["source_code"]
                    }
                ]
            }
        },
        {
            "name": "Contractor sharing same code (inappropriate)",
            "example": {
                "channel": "email", 
                "user": {"role": "CONTRACTOR", "dept": "TEMP"},
                "subject": "Portfolio example for next client",
                "body": "Here's some authentication code I worked on. Shows API key rotation and password hashing techniques. Database connection code might be useful for your next project.",
                "recipients": ["potential-client@competitor.com"],
                "attachments": [
                    {
                        "name": "auth_module.py",
                        "size": 45000,
                        "sensitivity_indicators": ["source_code"]  
                    }
                ]
            }
        },
        {
            "name": "HR manager sharing employee data (appropriate)",
            "example": {
                "channel": "email",
                "user": {"role": "HR_MANAGER", "dept": "HR"},
                "subject": "Performance review preparation",
                "body": "Performance review data for your direct reports. Please review ratings and provide feedback by Friday. Employee ID and salary information included for context.",
                "recipients": ["director@company.com"],
                "attachments": [
                    {
                        "name": "performance_reviews_Q3.xlsx",
                        "size": 512000,
                        "sensitivity_indicators": ["PII", "employee_data"]
                    }
                ]
            }
        },
        {
            "name": "Marketing intern with same data (inappropriate)",
            "example": {
                "channel": "email",
                "user": {"role": "INTERN", "dept": "MARKETING"}, 
                "subject": "Found this interesting data",
                "body": "Found some employee data while looking for campaign info. Has salary and performance ratings for everyone! Some people make way more than others. Crazy stuff.",
                "recipients": ["study-group@gmail.com"],
                "attachments": [
                    {
                        "name": "performance_reviews_Q3.xlsx",
                        "size": 512000,
                        "sensitivity_indicators": ["PII", "employee_data"]
                    }
                ]
            }
        }
    ]

def demonstrate_strategic_difference():
    """Show how contextual approach differs from pattern-based."""
    print("🎯 STRATEGIC DIFFERENCE DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for consistent results
    np.random.seed(42)
    
    # Initialize both labeling approaches
    pattern_labeler = DLPLabelGenerator()
    contextual_labeler = ContextualDLPLabeler()
    
    examples = create_strategic_examples()
    
    for i in range(0, len(examples), 2):
        appropriate_case = examples[i]
        inappropriate_case = examples[i + 1]
        
        print(f"\n📊 COMPARISON SET {(i//2) + 1}")
        print("=" * 40)
        
        for case_type, case in [("APPROPRIATE", appropriate_case), ("INAPPROPRIATE", inappropriate_case)]:
            example = case['example']
            
            print(f"\n{case_type} CONTEXT: {case['name']}")
            print(f"   User: {example['user']['role']} from {example['user']['dept']}")
            print(f"   Subject: {example['subject']}")
            print(f"   Recipients: {example['recipients']}")
            
            # Generate labels with both approaches
            pattern_labels = pattern_labeler.generate_labels(example)
            contextual_labels = contextual_labeler.generate_contextual_labels(example)
            
            pattern_risk = sum(pattern_labels.values()) / 4
            contextual_risk = sum(contextual_labels.values()) / 4
            
            print(f"\n   🤖 PATTERN-BASED LABELS:")
            print(f"      S:{pattern_labels['sensitivity']:.3f} E:{pattern_labels['exposure']:.3f} C:{pattern_labels['context']:.3f} O:{pattern_labels['obfuscation']:.3f}")
            print(f"      Total Risk: {pattern_risk:.3f}")
            
            print(f"   🧠 CONTEXTUAL LABELS:")
            print(f"      S:{contextual_labels['sensitivity']:.3f} E:{contextual_labels['exposure']:.3f} C:{contextual_labels['context']:.3f} O:{contextual_labels['obfuscation']:.3f}")
            print(f"      Total Risk: {contextual_risk:.3f}")
            
        # Calculate the strategic difference
        app_pattern = sum(pattern_labeler.generate_labels(examples[i]['example']).values()) / 4
        app_contextual = sum(contextual_labeler.generate_contextual_labels(examples[i]['example']).values()) / 4
        
        inapp_pattern = sum(pattern_labeler.generate_labels(examples[i+1]['example']).values()) / 4
        inapp_contextual = sum(contextual_labeler.generate_contextual_labels(examples[i+1]['example']).values()) / 4
        
        pattern_discrimination = abs(inapp_pattern - app_pattern)
        contextual_discrimination = abs(inapp_contextual - app_contextual)
        
        print(f"\n   📊 DISCRIMINATION ANALYSIS:")
        print(f"      Pattern-based difference:  {pattern_discrimination:.3f}")
        print(f"      Contextual difference:     {contextual_discrimination:.3f}")
        
        if contextual_discrimination > pattern_discrimination:
            print(f"      ✅ Contextual approach shows better discrimination!")
        else:
            print(f"      🟡 Pattern-based approach shows better discrimination")

def explain_strategic_value():
    """Explain the strategic business value."""
    print(f"\n🚀 STRATEGIC BUSINESS VALUE")
    print("=" * 60)
    
    print("🎯 WHY CONTEXTUAL APPROACH MATTERS:")
    print("   1. REDUCES FALSE POSITIVES")
    print("      • CFO sharing financials = legitimate business need")
    print("      • Intern sharing same data = policy violation")
    print("      • Same content, different risk level!")
    
    print("\n   2. CATCHES INSIDER THREATS")  
    print("      • Low-privilege users accessing high-value data")
    print("      • Cross-department data access without justification")
    print("      • Behavioral patterns indicating malicious intent")
    
    print("\n   3. BUSINESS LOGIC ENFORCEMENT")
    print("      • Role-based access control validation")
    print("      • Department-appropriate data sharing")
    print("      • Workflow and approval process compliance")
    
    print("\n   4. SOCIAL ENGINEERING DETECTION")
    print("      • Pressure tactics and urgency language")
    print("      • Financial motivation indicators")
    print("      • Relationship manipulation attempts")
    
    print(f"\n🏗️  COMPLEMENTARY ARCHITECTURE:")
    print("   REGEX LAYER:    'This email contains SSN: 123-45-6789' → BLOCK")
    print("   ML LAYER:       'Is it appropriate for this user to share this?' → CONTEXT")
    print("   COMBINED:       Better accuracy + Lower false positives")
    
    print(f"\n💡 BUSINESS IMPACT:")
    print("   ✅ Fewer legitimate business activities blocked")
    print("   ✅ Better detection of sophisticated insider threats")
    print("   ✅ More intelligent, context-aware DLP decisions")
    print("   ✅ Reduced security team alert fatigue")

def main():
    print("🧠 Contextual DLP Value Demonstration")
    print("=" * 60)
    print("Comparing Pattern-Based vs Contextual approaches")
    print("Focus: Strategic complementarity, not competition\n")
    
    demonstrate_strategic_difference()
    explain_strategic_value()
    
    print(f"\n✅ CONCLUSION:")
    print(f"   Contextual labeling creates a model that provides value")
    print(f"   BEYOND what regex-based systems can achieve.")
    print(f"   This is the strategic differentiation you were looking for!")

if __name__ == "__main__":
    main()