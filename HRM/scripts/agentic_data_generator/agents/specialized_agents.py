"""
Specialized agents for specific generation scenarios.
"""

from typing import Optional, Dict, List
from .base_agent import BaseLLMAgent, GenerationRequest, GeneratedExample
from utils.patterns import SpanDetector


class CasualAgent(BaseLLMAgent):
    """Casual domain specialist - Normal business communications."""
    
    def __init__(self, config):
        super().__init__(config, "Casual")
    
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate normal business communication that might accidentally contain sensitive information.

Scenario: {request.scenario_context}
Risk Level: {request.risk_level}
Required Spans: {request.target_spans}

Casual context ideas:
- Team updates and project communications
- Meeting notes and scheduling
- Customer support and external communications
- Sales and marketing outreach
- General workplace conversations

Generate JSON with this structure:
{{
  "channel": "email",
  "user": {{"role": "MARKETING", "dept": "BUSINESS", "seniority": "MANAGER"}},
  "recipients": ["team@company.com"],
  "subject": "Project update",
  "body": "Normal business communication that might accidentally contain sensitive information...",
  "attachments": ["project_notes.docx"],
  "links": ["https://company-tool.com/project"]
}}

Make it feel like normal workplace communication where sensitive info might slip in accidentally."""

        system_prompt = "You are simulating everyday workplace communication with natural, realistic business emails."
        
        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_casual_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
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
                    meta={"agent": "casual", **data.get("meta", {})}
                )
        
        self.generation_stats["failed"] += 1
        return None
    
    def _validate_casual_example(self, data: Dict) -> bool:
        """Validate casual communication requirements."""
        body = data.get("body", "")
        
        # Should feel like normal business communication
        business_terms = ["project", "team", "meeting", "update", "client", "deadline"]
        has_business_context = any(term in body.lower() for term in business_terms)
        
        # Avoid overly formal language
        overly_formal = ["pursuant to", "heretofore", "whereas", "indemnification"]
        not_overly_formal = not any(term in body.lower() for term in overly_formal)
        
        return has_business_context and not_overly_formal


class CleanBusinessAgent(BaseLLMAgent):
    """Specialist for generating clean business communications with NO sensitive information."""
    
    def __init__(self, config):
        super().__init__(config, "CleanBusiness")
    
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate completely clean, normal business communications with NO sensitive information.

Scenario: {request.scenario_context}
Risk Level: {request.risk_level}

CRITICAL REQUIREMENT: Generate realistic workplace communication that contains ZERO sensitive information:
- NO credit card numbers, SSNs, passwords, API keys, or secrets
- NO personal information beyond public business context
- NO confidential data or proprietary information
- Use only public, non-sensitive business information

Business scenarios for clean communication:
- Project status updates and general coordination
- Public meeting announcements and scheduling
- General company news and policy updates
- Standard vendor coordination (without credentials)

Generate JSON with this structure:
{{
  "channel": "email",
  "user": {{"role": "MARKETING", "dept": "BUSINESS", "seniority": "MANAGER"}},
  "recipients": ["team@company.com"],
  "subject": "Clean business subject", 
  "body": "Clean business communication with absolutely no sensitive information...",
  "attachments": ["public_proposal.pdf"],
  "links": ["https://public.company.com/info"]
}}

The communication must feel completely normal and contain zero sensitive information."""

        system_prompt = "Generate completely clean workplace communication with absolutely no sensitive information, personal data, secrets, or confidential content."
        
        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_clean_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
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
                    meta={"agent": "clean_business", **data.get("meta", {})}
                )
        
        self.generation_stats["failed"] += 1
        return None
    
    def _validate_clean_example(self, data: Dict) -> bool:
        """Validate that example contains NO sensitive information."""
        import re
        
        body = data.get("body", "")
        
        # Check for realistic business content
        business_terms = ["project", "team", "meeting", "update", "schedule", "review"]
        has_business_context = any(term in body.lower() for term in business_terms)
        
        # Ensure NO sensitive patterns
        sensitive_patterns = [
            r'\\b4[0-9]{15}\\b',  # Credit card
            r'\\b\\d{3}-?\\d{2}-?\\d{4}\\b',  # SSN
            r'\\bsk-[A-Za-z0-9]{20,}\\b',  # API key
            r'\\b[A-Za-z0-9]{20,}\\b',  # High entropy strings
        ]
        
        has_sensitive_patterns = any(re.search(pattern, body) for pattern in sensitive_patterns)
        
        # Ensure NO semantic obfuscation attempts
        semantic_indicators = ["one two three four", "dot", "dash", "encoded", "base64", "credential"]
        has_semantic_indicators = any(indicator in body.lower() for indicator in semantic_indicators)
        
        return has_business_context and not has_sensitive_patterns and not has_semantic_indicators


class ObfuscationSpecialist(BaseLLMAgent):
    """Specialist for sophisticated semantic obfuscation techniques."""
    
    def __init__(self, config):
        super().__init__(config, "ObfuscationSpecialist")
        self.span_detector = SpanDetector()
    
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        obfuscation_level = self._map_risk_to_sophistication(request.risk_level)
        
        prompt = f"""Generate sophisticated, semantically obfuscated communications that contain hidden sensitive information.

Scenario: {request.scenario_context}
Risk Level: {request.risk_level}
Required Spans: {request.target_spans}
Obfuscation Level: {obfuscation_level}

SEMANTIC OBFUSCATION TECHNIQUES:
1. NUMERICAL WORDS: "four five three two" instead of "4532"
2. TEXTUAL DECOMPOSITION: "john dot smith at company dot com" instead of "john.smith@company.com"  
3. EUPHEMISMS: "sixteen-digit payment identifier" instead of "credit card"
4. MULTILINGUAL: "cuatro cinco tres dos" (Spanish) instead of "4532"
5. CONTEXTUAL HIDING: "the authentication token from yesterday's call"
6. TECHNICAL REFERENCES: "base64 encoded client data in secure folder"
7. SPATIAL BREAKDOWN: "5 3 2 6   1 2 3 4   5 6 7 8"

Generate JSON with this structure:
{{
  "channel": "email",
  "user": {{"role": "SECURITY", "dept": "INFOSEC", "seniority": "ENGINEER"}},
  "recipients": ["colleague@company.com"],
  "subject": "Realistic subject that masks the sensitive content",
  "body": "Realistic business communication with semantically hidden sensitive information...",
  "attachments": ["config_backup.txt"],
  "links": ["https://company-portal.com/path"]
}}

The communication should feel natural while containing hidden sensitive data that only semantic analysis could detect."""

        system_prompt = f"""Create business communications that contain semantically obfuscated sensitive information.
The communication must feel natural and business-appropriate while hiding sensitive information using semantic techniques.
Focus on realism - this should feel like real workplace communication where someone is being clever about sharing sensitive information."""

        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_obfuscation_example(data, obfuscation_level):
                self.generation_stats["successful"] += 1
                
                techniques_used = self.span_detector.identify_obfuscation_techniques(data.get("body", ""))
                metadata = {
                    "agent": "obfuscation",
                    "sophistication_level": obfuscation_level,
                    "obfuscation_techniques": techniques_used
                }
                
                return GeneratedExample(
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
        
        self.generation_stats["failed"] += 1
        return None
    
    def _map_risk_to_sophistication(self, risk_level: str) -> str:
        """Map risk level to obfuscation sophistication."""
        mapping = {
            "low_risk": "low",
            "medium_risk": "medium", 
            "high_risk": "high",
            "obfuscated": "high"
        }
        return mapping.get(risk_level, "medium")
    
    def _validate_obfuscation_example(self, data: Dict, sophistication: str) -> bool:
        """Validate that example uses appropriate obfuscation techniques."""
        body = data.get("body", "")
        
        # Check for semantic obfuscation indicators
        techniques = self.span_detector.identify_obfuscation_techniques(body)
        
        # Require at least one obfuscation technique
        if sophistication == "high":
            return len(techniques) >= 2
        
        return len(techniques) >= 1