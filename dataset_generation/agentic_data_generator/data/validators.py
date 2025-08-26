"""
Quality validation for generated DLP training examples.
"""

from typing import Dict, List, Any
from utils.patterns import SpanDetector


class QualityValidator:
    """Validates quality of generated DLP examples."""
    
    def __init__(self):
        self.span_detector = SpanDetector()
        
    def validate_example(self, example: Any, min_score: float = 0.7) -> bool:
        """Validate if example meets quality requirements."""
        score = self.calculate_quality_score(example)
        return score >= min_score
    
    def calculate_quality_score(self, example: Any) -> float:
        """Calculate comprehensive quality score (0-1)."""
        score = 0.0
        
        # Content quality (40%)
        if hasattr(example, 'body') and example.body:
            body = example.body
            
            # Length check (0.2)
            if 50 <= len(body) <= 2000:
                score += 0.2
            
            # Content realism (0.2)
            if self._has_realistic_content(body):
                score += 0.2
        
        # Structure quality (30%)
        if self._has_valid_structure(example):
            score += 0.3
            
        # Span quality (30%)
        if hasattr(example, 'body'):
            span_score = self._evaluate_spans(example.body)
            score += span_score * 0.3
        
        return min(score, 1.0)
    
    def _has_realistic_content(self, body: str) -> bool:
        """Check if content feels realistic."""
        # Avoid placeholder text
        placeholders = ["lorem", "placeholder", "example", "test", "dummy"]
        if any(word in body.lower() for word in placeholders):
            return False
            
        # Should contain business-like terms
        business_terms = [
            "project", "meeting", "client", "team", "update", "review",
            "payment", "invoice", "contract", "agreement", "schedule",
            "employee", "benefits", "policy", "security", "access"
        ]
        return any(term in body.lower() for term in business_terms)
    
    def _has_valid_structure(self, example: Any) -> bool:
        """Check if example has valid structure."""
        checks = []
        
        # Subject check
        if hasattr(example, 'subject') and example.subject:
            checks.append(len(example.subject) > 5)
        else:
            checks.append(False)
            
        # Recipients check
        if hasattr(example, 'recipients') and example.recipients:
            checks.append(all("@" in r for r in example.recipients))
        else:
            checks.append(False)
            
        # User info check
        if hasattr(example, 'user') and isinstance(example.user, dict):
            checks.append('role' in example.user)
        else:
            checks.append(False)
            
        return sum(checks) >= 2  # At least 2 out of 3 structure checks
    
    def _evaluate_spans(self, body: str) -> float:
        """Evaluate span detection quality."""
        spans = self.span_detector.extract_all_spans(body)
        
        if len(spans) == 0:
            return 0.3  # Clean content is okay, but not optimal for training
        
        if 1 <= len(spans) <= 5:
            return 1.0  # Good span count
            
        if len(spans) > 5:
            return 0.7  # Too many spans might be unrealistic
            
        return 0.5
    
    def validate_agent_specific(self, example: Any, agent_type: str) -> bool:
        """Validate agent-specific requirements."""
        if not hasattr(example, 'body'):
            return False
            
        body = example.body.lower()
        
        validation_rules = {
            "legal": {
                "required_terms": ["matter", "nda", "agreement", "confidential", "privilege", "counsel"],
                "required_roles": ["LEGAL", "COMPLIANCE", "PRIVACY"]
            },
            "finance": {
                "required_terms": ["payment", "invoice", "credit", "bank", "account", "wire", "ach", "routing"],
                "required_roles": ["FINANCE", "ACCOUNTING", "TREASURY", "PAYMENTS"]
            },
            "hr": {
                "required_terms": ["employee", "onboarding", "benefits", "performance", "salary", "personal", "hire"],
                "required_roles": ["HR", "PEOPLE", "RECRUITING", "BENEFITS"]
            },
            "security": {
                "required_terms": ["credential", "api", "key", "secret", "token", "access", "vault", "security"],
                "required_roles": ["SECURITY", "DEVOPS", "INFOSEC", "SRE"]
            }
        }
        
        if agent_type not in validation_rules:
            return True  # No specific rules for other agents
            
        rules = validation_rules[agent_type]
        
        # Check for required terms
        has_terms = any(term in body for term in rules["required_terms"])
        
        # Check user role
        has_role = False
        if hasattr(example, 'user') and isinstance(example.user, dict):
            user_role = example.user.get('role', '')
            has_role = user_role in rules["required_roles"]
        
        return has_terms and has_role