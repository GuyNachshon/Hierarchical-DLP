#!/usr/bin/env python3
"""
Test script for attachment schema fixes.
Tests the updated domain agents, validation functions, and structured output.
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_data_generator.agents.domain_agents import LegalAgent, FinanceAgent, HRAgent, SecurityAgent
from agentic_data_generator.agents.base_agent import GenerationRequest


class TestConfig:
    """Simple config for testing."""
    def __init__(self):
        self.temperature = 0.8
        self.max_retries = 2


async def test_domain_agent(agent_class, agent_type, scenario):
    """Test a single domain agent for proper attachment generation."""
    print(f"\n=== Testing {agent_type} Agent ===")
    
    config = TestConfig()
    agent = agent_class(config)
    
    request = GenerationRequest(
        agent_type=agent_type,
        risk_level="medium",
        scenario_context=scenario,
        target_spans=["credentials", "pii", "confidential"],
        conversation_turns=1,
        count=1
    )
    
    try:
        example = await agent.generate_example(request)
        
        if example:
            print(f"✅ Generation successful!")
            print(f"Subject: {example.subject}")
            print(f"Attachments: {len(example.attachments)} found")
            
            # Validate attachment structure
            for i, attachment in enumerate(example.attachments):
                print(f"\nAttachment {i+1}:")
                if isinstance(attachment, dict):
                    required_fields = ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"]
                    missing_fields = [field for field in required_fields if field not in attachment]
                    
                    if missing_fields:
                        print(f"❌ Missing fields: {missing_fields}")
                        return False
                    else:
                        print(f"✅ All required fields present")
                        print(f"   Name: {attachment['name']}")
                        print(f"   Size: {attachment['size']}")
                        print(f"   MIME: {attachment['mime_type']}")
                        print(f"   Summary: {attachment['content_summary'][:50]}...")
                        print(f"   Indicators: {attachment['sensitivity_indicators']}")
                else:
                    print(f"❌ Attachment is not a dict: {type(attachment)}")
                    return False
                    
            return True
        else:
            print(f"❌ Generation failed - no example returned")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed with error: {e}")
        return False


async def main():
    """Run tests for all domain agents."""
    print("Testing Attachment Schema Fixes")
    print("=" * 40)
    
    tests = [
        (LegalAgent, "legal", "NDA review with external counsel"),
        (FinanceAgent, "finance", "Payment processing with bank partner"),
        (HRAgent, "hr", "Employee onboarding with personal data"),
        (SecurityAgent, "security", "API key sharing for deployment")
    ]
    
    results = []
    for agent_class, agent_type, scenario in tests:
        success = await test_domain_agent(agent_class, agent_type, scenario)
        results.append((agent_type, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for agent_type, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{agent_type.upper():10} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Attachment schema fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())