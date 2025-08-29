#!/usr/bin/env python3
"""
Quick HRM-DLP Model Test

Simple test script to quickly validate model predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from test_model import HRMDLPTester

def main():
    print("🚀 HRM-DLP Quick Test")
    print("=" * 50)
    
    # Initialize tester with latest checkpoint
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    try:
        tester = HRMDLPTester(checkpoint_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test scenarios
    scenarios = [
        {
            "name": "🔴 HIGH RISK - Financial data to external email",
            "example": {
                "channel": "email",
                "user": {"role": "FINANCE", "dept": "FINANCE"},
                "recipients": ["external@gmail.com"],
                "subject": "CONFIDENTIAL - Q3 Revenue Report", 
                "body": "Our Q3 revenue was $5.2M. Customer SSN for largest client: 123-45-6789. Please review the attached financial statements and wire transfer details: Account 1234-5678-9012-3456.",
                "attachments": [
                    {"name": "revenue_report.xlsx", "size": 256000, "mime": "application/vnd.ms-excel"}
                ]
            }
        },
        {
            "name": "🟡 MEDIUM RISK - Internal document with some PII",
            "example": {
                "channel": "email",
                "user": {"role": "HR", "dept": "HR"},
                "recipients": ["manager@company.com"],
                "subject": "Employee Performance Review",
                "body": "John Smith's performance review is complete. His employee ID is EMP-12345. Contact number: 555-123-4567. Overall rating: Exceeds Expectations.",
                "attachments": []
            }
        },
        {
            "name": "🟢 LOW RISK - General business communication",
            "example": {
                "channel": "email",
                "user": {"role": "MARKETING", "dept": "MARKETING"}, 
                "recipients": ["team@company.com"],
                "subject": "Weekly Team Update",
                "body": "Hi team, here's our weekly update. Project Alpha is on track for Q4 launch. Meeting scheduled for Tuesday at 2 PM in Conference Room B.",
                "attachments": []
            }
        },
        {
            "name": "🔍 OBFUSCATION - Suspicious encoding patterns",
            "example": {
                "channel": "email",
                "user": {"role": "ENG", "dept": "ENGINEERING"},
                "recipients": ["contractor@external.com"],
                "subject": "Database Setup",
                "body": "Connection details: cG9zdGdyZXM6Ly91c2VyOnBAc3N3MHJkQGRiLmNvbXBhbnkuY29tOjU0MzI= (base64 encoded for security). Also, API key: ak_1234567890abcdef. Use asterisk masking for sensitive data like ***-**-1234.",
                "attachments": []
            }
        }
    ]
    
    print(f"🧪 Running {len(scenarios)} test scenarios...\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{scenario['name']}")
        print("-" * 60)
        
        try:
            result = tester.predict_single(scenario['example'], verbose=True)
            print(f"✅ Scenario {i} completed successfully\n")
            
        except Exception as e:
            print(f"❌ Scenario {i} failed: {e}\n")
    
    print("🎯 Quick test completed!")
    print("\n💡 To run more comprehensive tests:")
    print("   python test_model.py --synthetic")
    print("   python test_model.py --dataset data/hrm_dlp_final/test.jsonl --max_examples 5")

if __name__ == "__main__":
    main()