import re
from typing import Optional, Dict

from config import AgenticConfig
from agents.base_agent import GenerationRequest, GeneratedExample, BaseLLMAgent


class AugmentationAgent(BaseLLMAgent):
    """Augmentation Agent - Flips scenarios between leak/legit cases"""

    def __init__(self, config: AgenticConfig):
        super().__init__(config, "Augmentation")

    async def flip_scenario(self, example: GeneratedExample, target_risk: str) -> Optional[GeneratedExample]:
        """Flip an example from leak->legit or legit->leak while preserving sensitive data"""

        current_risk = self._infer_current_risk_level(example)

        if target_risk == current_risk:
            return example  # No change needed

        flip_prompt = f"""You are an expert at transforming DLP scenarios by flipping the context between legitimate and violation cases.

ORIGINAL EXAMPLE:
Channel: {example.channel}
User: {example.user}
Recipients: {example.recipients}
Subject: {example.subject}
Body: {example.body}
Current Risk Level: {current_risk}

TASK: Transform this to a {target_risk} scenario by changing ONLY the context/recipients/setting while preserving ALL sensitive data.

TRANSFORMATION RULES:
1. PRESERVE ALL SENSITIVE DATA: Keep exact same credit cards, SSNs, passwords, API keys, emails, names, etc.
2. CHANGE CONTEXT ONLY: Modify recipients, business justification, and authorization level
3. MAINTAIN REALISM: The transformed scenario must feel completely natural. emails , names, content etc. should be realistic

FLIP STRATEGIES:

FOR LEAK → LEGIT TRANSFORMATION:
- Change personal emails (gmail.com) to authorized business domains (law firms, banks, vendors)
- Add proper business context and authorization
- Include appropriate business relationships
- Example: "john@gmail.com" → "counsel@smithlaw.com" with "per our retainer agreement"

FOR LEGIT → LEAK TRANSFORMATION:
- Change authorized recipients to unauthorized ones (personal emails, wrong domains)
- Remove business justification 
- Create inappropriate sharing scenarios
- Example: "billing@vendor.com" → "shlomo@gmail.com" with casual context

SPECIFIC DOMAIN GUIDANCE:
- Legal: Use law firm domains (counsel@lawfirm.com) for legit, personal emails for leaks
- Finance: Use bank/vendor domains (billing@bank.com) for legit, personal for leaks
- HR: Use authorized vendors (hr@benefits.com) for legit, personal for leaks
- Security: Use authorized tools (admin@vault.com) for legit, casual sharing for leaks

Generate the transformed example in JSON format:
{{
  "channel": "{example.channel}",
  "user": {example.user},
  "recipients": ["new_recipient@appropriate_domain.com"],
  "subject": "Transformed subject maintaining context",
  "body": "Transformed body with new context but EXACT same sensitive data...",
  "attachments": {example.attachments},
  "links": ["updated_links_if_needed"]
}}

The sensitive information must be IDENTICAL. Only change the business context and authorization level."""

        system_prompt = f"""You are an expert at DLP scenario transformation. Your job is to flip examples between legitimate and violation cases while preserving all sensitive data exactly.

Key principles:
1. Never modify sensitive data (credit cards, SSNs, passwords, etc.)
2. Only change business context and authorization
3. Make transformations feel natural and realistic
4. Understand the difference between authorized and unauthorized sharing"""

        response = await self._generate_with_llm(flip_prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_flip_transformation(example, data):
                # Create new example with augmentation metadata
                metadata = {
                    "agent": "augmentation",
                    "original_agent": example.generation_metadata.get("agent") if example.generation_metadata else "unknown",
                    "flip_direction": f"{current_risk} → {target_risk}",
                    "original_risk": current_risk,
                    "target_risk": target_risk
                }
                flipped_example = GeneratedExample(
                    channel=data.get("channel", "email"),
                    user=data.get("user", {}),
                    recipients=data.get("recipients", []),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    attachments=data.get("attachments", []),
                    links=data.get("links", []),
                    thread=data.get("thread"),
                    labels=data.get("labels"),
                    spans=data.get("spans"),
                    meta=metadata
                )
                return flipped_example

        return None

    # Helper used by batch augmentation paths
    def build_flip_prompts(self, example: GeneratedExample, target_risk: str) -> Dict[str, str]:
        current_risk = self._infer_current_risk_level(example)
        system_prompt = (
            "You are an expert at DLP scenario transformation. Your job is to flip examples between "
            "legitimate and violation cases while preserving all sensitive data exactly.\n"
            "Rules: Never modify sensitive data; change only context/recipients/authorization; "
            "reply strictly in JSON only."
        )
        flip_prompt = f"""ORIGINAL EXAMPLE:
Channel: {example.channel}
User: {example.user}
Recipients: {example.recipients}
Subject: {example.subject}
Body: {example.body}
Current Risk Level: {current_risk}

Transform to: {target_risk}

Constraints:
- Preserve ALL sensitive values exactly (cards, SSNs, passwords, keys, names, emails text)
- Change only business context, recipients, domains, justification
- Make it realistic and coherent

Return JSON only with keys: channel, user, recipients, subject, body, attachments, links
"""
        return {"system": system_prompt, "user": flip_prompt}

    def _infer_current_risk_level(self, example: GeneratedExample) -> str:
        """Infer current risk level from example content and recipients"""

        # Check recipients for personal domains (high risk indicators)
        personal_domains = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "proton.me", "aol.com"}
        external_personal = any(any(domain in recipient for domain in personal_domains)
                                for recipient in example.recipients)

        # Check for business domains and authorized contexts
        has_business_context = any(word in example.body.lower() for word in
                                   ["authorized", "agreement", "retainer", "contract", "vendor", "partner"])

        # Check for casual language indicating leak
        has_casual_language = any(phrase in example.body.lower() for phrase in
                                  ["hey", "fyi", "just sharing", "thought you'd want", "quick note"])

        # Simple risk inference
        if external_personal and not has_business_context:
            return "high_risk"
        elif has_casual_language:
            return "medium_risk"
        elif has_business_context:
            return "low_risk"
        else:
            return "medium_risk"

    def _validate_flip_transformation(self, original: GeneratedExample, flipped_data: Dict) -> bool:
        """Validate that flip preserves sensitive data while changing context"""

        original_body = original.body
        flipped_body = flipped_data.get("body", "")

        # Extract sensitive patterns from both
        sensitive_patterns = [
            r'\b4[0-9]{15}\b',  # Credit card
            r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
            r'\bsk-[A-Za-z0-9]{20,}\b',  # API key
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
        ]

        original_sensitive = []
        flipped_sensitive = []

        for pattern in sensitive_patterns:
            original_sensitive.extend(re.findall(pattern, original_body, re.IGNORECASE))
            flipped_sensitive.extend(re.findall(pattern, flipped_body, re.IGNORECASE))

        # Check that key sensitive data is preserved
        # Allow some variation in emails (different domains is expected)
        original_non_email = [x for x in original_sensitive if "@" not in x]
        flipped_non_email = [x for x in flipped_sensitive if "@" not in x]

        # Non-email sensitive data should be identical
        return len(original_non_email) > 0 and set(original_non_email) == set(flipped_non_email)

    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        """Not used - augmentation agent works on existing examples"""
        return None
