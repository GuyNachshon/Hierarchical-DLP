#!/usr/bin/env python3
"""
LLM-Based DLP Label Generation Strategy

Uses OpenAI/Anthropic models to generate high-quality, contextually-aware 
DLP risk labels that focus on business appropriateness rather than pattern matching.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class LLMLabelingConfig:
    """Configuration for LLM-based labeling."""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4o"     # or "claude-3-5-sonnet-20241022"
    temperature: float = 0.1  # Low temperature for consistent labeling
    max_tokens: int = 200     # Enough for structured JSON response
    batch_size: int = 10      # Process in batches to manage API limits
    rate_limit_delay: float = 1.0  # Seconds between API calls

class DLPPromptStrategy:
    """
    Sophisticated prompting strategy for LLM-based DLP labeling.
    
    Key Philosophy: Focus on business context and appropriateness,
    NOT pattern detection (which regex handles better).
    """
    
    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.few_shot_examples = self._create_few_shot_examples()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt that defines the LLM's role and focus."""
        return """You are an expert Data Loss Prevention (DLP) analyst specializing in CONTEXTUAL RISK ASSESSMENT for business communications.

Your job is to assess the BUSINESS APPROPRIATENESS of email communications, NOT to detect patterns (regex handles that better).

Focus on these key dimensions:

1. SENSITIVITY RISK (0.0-1.0): Business context appropriateness
   - Is this data sharing appropriate for the business context?
   - Does the content match legitimate business needs?
   - Consider: strategic importance, business confidentiality, competitive sensitivity
   - HIGH: Trade secrets, M&A info, strategic plans shared inappropriately
   - LOW: Routine business updates, public information, approved disclosures

2. EXPOSURE RISK (0.0-1.0): Recipient relationship appropriateness  
   - Are these recipients appropriate for this content?
   - Do sender-recipient relationships make business sense?
   - Consider: business justification, organizational boundaries, external risks
   - HIGH: Sensitive info to competitors, personal emails for business data
   - LOW: Appropriate departmental sharing, authorized external communications

3. CONTEXT RISK (0.0-1.0): Situational and behavioral appropriateness
   - Does the user's role/department justify access to this information?
   - Are there behavioral red flags or unusual patterns?
   - Consider: role-based access, cross-departmental boundaries, user trustworthiness
   - HIGH: Low-level users accessing high-value data, unauthorized cross-department sharing
   - LOW: Appropriate role-based access, normal business workflows

4. OBFUSCATION RISK (0.0-1.0): Intent and social engineering indicators
   - Are there signs of deceptive intent or social engineering?
   - Does the communication style suggest manipulation or coercion?
   - Consider: pressure tactics, urgency manipulation, relationship exploitation
   - HIGH: Social engineering, financial incentives, pressure tactics
   - LOW: Transparent business communication, normal professional tone

CRITICAL: Your analysis should complement, not compete with, regex-based pattern detection.
Focus on CONTEXT and APPROPRIATENESS, not obvious patterns like SSNs or credit cards."""

    def _create_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Create few-shot examples to guide the LLM's reasoning."""
        return [
            {
                "input": {
                    "user_role": "CFO",
                    "user_dept": "FINANCE", 
                    "recipients": ["board-secretary@company.com", "ceo@company.com"],
                    "subject": "Q4 Board Presentation - Financial Summary",
                    "body": "Attached are Q4 financials for board meeting. Revenue $50M, EBITDA $12M. Executive compensation details included for discussion.",
                    "attachments": [{"name": "Q4_Board_Financials.xlsx", "size": 2048000}],
                    "business_context": "Board meeting preparation by senior executive"
                },
                "reasoning": {
                    "sensitivity": "Low (0.2) - Appropriate business context: CFO sharing financials for legitimate board meeting",
                    "exposure": "Very Low (0.1) - Perfect recipients: Board secretary and CEO are appropriate for financial data", 
                    "context": "Very Low (0.05) - Ideal context: Senior executive in appropriate role sharing with authorized recipients",
                    "obfuscation": "Very Low (0.0) - Transparent business communication with clear purpose"
                },
                "output": {
                    "sensitivity": 0.2,
                    "exposure": 0.1, 
                    "context": 0.05,
                    "obfuscation": 0.0
                }
            },
            {
                "input": {
                    "user_role": "INTERN",
                    "user_dept": "MARKETING",
                    "recipients": ["friend@gmail.com", "roommate@yahoo.com"],
                    "subject": "Crazy salary info I found!",
                    "body": "Found this financial data on shared drive. Revenue $50M, executives make crazy money! Check out these salaries - CEO gets $2M bonus!",
                    "attachments": [{"name": "Q4_Board_Financials.xlsx", "size": 2048000}],
                    "business_context": "Intern sharing confidential financial data to personal contacts"
                },
                "reasoning": {
                    "sensitivity": "Very High (0.95) - Highly inappropriate: Financial data shared outside business context",
                    "exposure": "Very High (0.9) - Terrible recipients: Personal Gmail/Yahoo accounts are inappropriate for business data",
                    "context": "Very High (0.85) - Major violation: Intern has no business justification for accessing/sharing financial data",
                    "obfuscation": "Moderate (0.4) - Casual tone suggests lack of awareness, not malicious intent"
                },
                "output": {
                    "sensitivity": 0.95,
                    "exposure": 0.9,
                    "context": 0.85, 
                    "obfuscation": 0.4
                }
            },
            {
                "input": {
                    "user_role": "HR_MANAGER", 
                    "user_dept": "HR",
                    "recipients": ["director@company.com"],
                    "subject": "Performance review preparation",
                    "body": "Performance review data for your direct reports. Please review ratings and feedback by Friday. Salary information included for context.",
                    "attachments": [{"name": "performance_reviews_Q3.xlsx", "size": 512000}],
                    "business_context": "HR manager sharing employee performance data with appropriate manager"
                },
                "reasoning": {
                    "sensitivity": "Low (0.25) - Appropriate context: HR sharing performance data with direct manager",
                    "exposure": "Very Low (0.1) - Good recipient: Director receiving data about their direct reports",
                    "context": "Very Low (0.1) - Perfect context: HR manager in appropriate role sharing with authorized recipient",
                    "obfuscation": "Very Low (0.05) - Professional, transparent communication"
                },
                "output": {
                    "sensitivity": 0.25,
                    "exposure": 0.1,
                    "context": 0.1,
                    "obfuscation": 0.05
                }
            }
        ]
    
    def create_labeling_prompt(self, email_data: Dict[str, Any]) -> str:
        """Create a complete prompt for labeling a single email."""
        
        # Extract key information
        user_info = email_data.get('user', {})
        user_role = user_info.get('role', 'UNKNOWN')
        user_dept = user_info.get('dept', 'UNKNOWN')
        
        recipients = email_data.get('recipients', [])
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        attachments = email_data.get('attachments', [])
        
        # Create structured input
        email_analysis = {
            "user_role": user_role,
            "user_dept": user_dept,
            "recipients": recipients,
            "subject": subject,
            "body": body[:1000] + ("..." if len(body) > 1000 else ""),  # Truncate long bodies
            "attachments": [
                {
                    "name": att.get('name', ''),
                    "size": att.get('size', 0),
                    "type": att.get('mime_type', att.get('mime', ''))
                }
                for att in attachments[:3]  # Limit to 3 attachments
            ] if attachments else []
        }
        
        prompt = f"""Analyze this email communication for DLP risk assessment:

EMAIL TO ANALYZE:
{json.dumps(email_analysis, indent=2)}

Provide your analysis focusing on BUSINESS APPROPRIATENESS and CONTEXT, not pattern detection.

Consider:
1. Is this sharing appropriate for the business context?
2. Are the recipients appropriate for this user and content?
3. Does the user's role justify access to this information?  
4. Are there any social engineering or manipulation indicators?

Respond with ONLY a JSON object in this exact format:
{{
  "sensitivity": 0.X,
  "exposure": 0.X, 
  "context": 0.X,
  "obfuscation": 0.X,
  "reasoning": {{
    "sensitivity": "Brief explanation of business context appropriateness",
    "exposure": "Brief explanation of recipient appropriateness", 
    "context": "Brief explanation of user role appropriateness",
    "obfuscation": "Brief explanation of intent/manipulation indicators"
  }}
}}

Remember: Focus on APPROPRIATENESS and CONTEXT, not obvious patterns that regex can detect."""

        return prompt
    
    def parse_llm_response(self, response: str) -> Optional[Dict[str, float]]:
        """Parse LLM response and extract labels."""
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Handle cases where LLM includes extra text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Extract just the numeric labels
            labels = {}
            for key in ['sensitivity', 'exposure', 'context', 'obfuscation']:
                if key in parsed and isinstance(parsed[key], (int, float)):
                    # Ensure values are in [0, 1] range
                    labels[key] = max(0.0, min(1.0, float(parsed[key])))
                else:
                    return None  # Missing or invalid label
            
            return labels
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

def demonstrate_prompt_strategy():
    """Demonstrate the prompt strategy with sample data."""
    print("🤖 LLM-Based DLP Labeling Strategy Demo")
    print("=" * 60)
    
    strategy = DLPPromptStrategy()
    
    # Sample email data
    sample_email = {
        "user": {"role": "CONTRACTOR", "dept": "TEMP"},
        "recipients": ["potential-client@competitor.com"],
        "subject": "Portfolio example for next client",
        "body": "Here's some authentication code I worked on. Shows API key rotation and password hashing techniques. Database connection code might be useful for your next project.",
        "attachments": [
            {"name": "auth_module.py", "size": 45000, "mime": "text/x-python"}
        ]
    }
    
    print("📧 SAMPLE EMAIL:")
    print(json.dumps(sample_email, indent=2))
    
    print(f"\n🎯 GENERATED PROMPT:")
    prompt = strategy.create_labeling_prompt(sample_email)
    print(prompt)
    
    print(f"\n💡 KEY ADVANTAGES OF LLM APPROACH:")
    print("   ✅ Understands business context and appropriateness")
    print("   ✅ Recognizes user role vs content appropriateness")  
    print("   ✅ Analyzes recipient relationships and justifications")
    print("   ✅ Detects intent and social engineering patterns")
    print("   ✅ Provides consistent, high-quality labels at scale")
    print("   ✅ Focuses on context vs patterns (complements regex)")

if __name__ == "__main__":
    demonstrate_prompt_strategy()