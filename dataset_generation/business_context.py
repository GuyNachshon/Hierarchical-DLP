#!/usr/bin/env python3
"""
Business Relationship Context System for DLP

This module provides sophisticated business context analysis to distinguish between:
1. Legitimate external sharing (authorized business relationships)
2. Inappropriate external sharing (unauthorized domains, personal emails)

Key insight: The same sensitive data can be appropriate or inappropriate 
depending entirely on business context and authorized relationships.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class DomainType(Enum):
    """Classification of domain types for business context"""
    PERSONAL = "personal"           # gmail, outlook, yahoo - personal email
    LAW_FIRM = "law_firm"          # Legal counsel, law firms
    FINANCIAL = "financial"        # Banks, payment processors, financial services
    CONTRACTOR = "contractor"      # Technical contractors, consultants
    VENDOR = "vendor"              # Business vendors, suppliers
    GOVERNMENT = "government"      # Government agencies, regulatory bodies
    INTERNAL = "internal"          # Same company domain
    SUSPICIOUS = "suspicious"      # Short domains, non-business patterns
    UNKNOWN = "unknown"            # Cannot classify


@dataclass
class BusinessRelationship:
    """Defines authorized business relationships for data sharing"""
    data_types: Set[str]           # What data types can be shared
    authorized_domains: Set[str]   # Specific authorized domains
    domain_patterns: List[str]     # Regex patterns for authorized domain types
    role_requirements: Set[str]    # User roles that can authorize this sharing
    context_keywords: Set[str]     # Content keywords that indicate legitimate context


class BusinessContextAnalyzer:
    """Analyzes business context to determine if external sharing is appropriate"""
    
    def __init__(self):
        self.domain_classifiers = self._init_domain_classifiers()
        self.business_relationships = self._init_business_relationships()
    
    def _init_domain_classifiers(self) -> Dict[DomainType, List[str]]:
        """Initialize domain classification patterns"""
        return {
            DomainType.PERSONAL: [
                r'gmail\.com$', r'outlook\.com$', r'yahoo\.com$', r'hotmail\.com$',
                r'proton\.me$', r'icloud\.com$', r'aol\.com$'
            ],
            
            DomainType.LAW_FIRM: [
                r'.*law\.com$', r'.*legal\.com$', r'.*law\.net$', r'.*legal\.net$',
                r'.*attorneys\.com$', r'.*counsel\.com$', r'.*\-law\.com$',
                r'.*lawfirm\.com$', r'.*\-legal\.com$', r'.*lawyers\.com$',
                r'.*\.law$', r'.*\.legal$', r'.*llp\.com$', r'.*\-llp\.com$'
            ],
            
            DomainType.FINANCIAL: [
                r'.*bank\.com$', r'.*banking\.com$', r'.*financial\.com$',
                r'.*payment\.com$', r'.*payments\.com$', r'.*finance\.com$',
                r'.*credit\.com$', r'.*lending\.com$', r'.*treasury\.com$',
                r'chase\.com$', r'wellsfargo\.com$', r'bankofamerica\.com$',
                r'jpmorgan\.com$', r'citi\.com$', r'stripe\.com$', r'paypal\.com$'
            ],
            
            DomainType.CONTRACTOR: [
                r'.*consulting\.com$', r'.*contractor\.com$', r'.*tech\.com$',
                r'.*solutions\.com$', r'.*services\.com$', r'.*consulting\.net$'
            ],
            
            DomainType.VENDOR: [
                r'.*vendor\.com$', r'.*supplier\.com$', r'.*corp\.com$',
                r'.*company\.com$', r'.*inc\.com$', r'.*ltd\.com$'
            ],
            
            DomainType.GOVERNMENT: [
                r'.*\.gov$', r'.*\.mil$', r'.*regulatory\.com$',
                r'.*compliance\.com$', r'.*audit\.com$'
            ],
            
            DomainType.SUSPICIOUS: [
                r'^[a-z]{1,4}\.com$',     # Very short domains
                r'^[0-9]+\.com$',         # Numeric domains
                r'.*temp.*\.com$',        # Temporary-sounding
                r'.*test.*\.com$'         # Test domains
            ]
        }
    
    def _init_business_relationships(self) -> Dict[str, BusinessRelationship]:
        """Initialize authorized business relationships"""
        return {
            "legal_external": BusinessRelationship(
                data_types={"NDA", "MATTER", "NAME", "EMAIL", "NDA_EUPHEMISM"},
                authorized_domains=set(),  # Will be pattern-matched
                domain_patterns=[DomainType.LAW_FIRM.value],
                role_requirements={"LEGAL", "COMPLIANCE", "PRIVACY"},
                context_keywords={"nda", "agreement", "matter", "counsel", "legal", "contract"}
            ),
            
            "financial_external": BusinessRelationship(
                data_types={"PAN", "EMAIL", "NAME", "PAN_SEMANTIC", "PAN_EUPHEMISM"},
                authorized_domains=set(),
                domain_patterns=[DomainType.FINANCIAL.value],
                role_requirements={"FINANCE", "ACCOUNTING", "TREASURY", "PAYMENTS"},
                context_keywords={"payment", "invoice", "bank", "wire", "processing", "vendor"}
            ),
            
            "hr_external": BusinessRelationship(
                data_types={"SSN", "NAME", "EMAIL", "PHONE", "SSN_SEMANTIC"},
                authorized_domains=set(),
                domain_patterns=[DomainType.VENDOR.value],  # Benefits providers, etc.
                role_requirements={"HR", "PEOPLE", "BENEFITS"},
                context_keywords={"benefits", "enrollment", "provider", "insurance"}
            ),
            
            "contractor_technical": BusinessRelationship(
                data_types={"EMAIL", "NAME", "DBURI"},  # Limited tech data
                authorized_domains=set(),
                domain_patterns=[DomainType.CONTRACTOR.value],
                role_requirements={"ENGINEERING", "DEVOPS", "TECH"},
                context_keywords={"project", "development", "contractor", "consulting"}
            )
        }
    
    def classify_domain(self, domain: str) -> DomainType:
        """Classify a domain into business context type"""
        domain_lower = domain.lower()
        
        for domain_type, patterns in self.domain_classifiers.items():
            for pattern in patterns:
                if re.match(pattern, domain_lower):
                    return domain_type
        
        return DomainType.UNKNOWN
    
    def is_authorized_external_sharing(self, 
                                     sensitive_spans: List[Dict],
                                     user_role: str,
                                     recipient_domains: List[str],
                                     content: str) -> bool:
        """Determine if external sharing is authorized based on business context"""
        
        content_lower = content.lower()
        
        # Extract span types present
        span_types = {span["type"] for span in sensitive_spans}
        
        # Check each business relationship
        for relationship_name, relationship in self.business_relationships.items():
            
            # 1. Check if user role is authorized for this relationship
            if user_role not in relationship.role_requirements:
                continue
            
            # 2. Check if content has appropriate business context keywords
            has_business_context = any(keyword in content_lower for keyword in relationship.context_keywords)
            if not has_business_context:
                continue
            
            # 3. Check if data types align with relationship
            data_overlap = span_types.intersection(relationship.data_types)
            if not data_overlap:
                continue
            
            # 4. Check if recipient domains are appropriate
            for domain in recipient_domains:
                domain_type = self.classify_domain(domain)
                
                # Check if domain type matches authorized patterns
                if domain_type.value in relationship.domain_patterns:
                    return True  # Found authorized external sharing
                
                # Check specific authorized domains
                if domain.lower() in relationship.authorized_domains:
                    return True
        
        return False  # No authorized relationship found
    
    def analyze_sharing_context(self, 
                              sensitive_spans: List[Dict],
                              user_role: str, 
                              recipients: List[str],
                              content: str,
                              sender_domain: Optional[str] = None) -> Dict[str, any]:
        """Comprehensive business context analysis"""
        
        recipient_domains = [r.split("@")[-1] for r in recipients if "@" in r]
        
        # Classify all recipient domains
        domain_classifications = {domain: self.classify_domain(domain) for domain in recipient_domains}
        
        # Check for personal domains (always high exposure risk)
        has_personal_domains = any(dt == DomainType.PERSONAL for dt in domain_classifications.values())
        
        # Check for authorized external sharing
        has_authorized_external = self.is_authorized_external_sharing(
            sensitive_spans, user_role, recipient_domains, content
        )
        
        # Check for internal sharing
        has_internal_only = sender_domain and all(domain == sender_domain for domain in recipient_domains)
        
        # Determine final labels
        exposure_risk = has_personal_domains or (recipient_domains and not has_authorized_external and not has_internal_only)
        legitimate_context = has_authorized_external or has_internal_only
        
        return {
            "exposure": 1 if exposure_risk else 0,
            "context": 1 if legitimate_context else 0,
            "analysis": {
                "recipient_domains": recipient_domains,
                "domain_classifications": {d: dt.value for d, dt in domain_classifications.items()},
                "has_personal_domains": has_personal_domains,
                "has_authorized_external": has_authorized_external,
                "has_internal_only": has_internal_only,
                "business_relationship_found": has_authorized_external
            }
        }
    
    def get_authorized_scenarios_for_agent(self, agent_type: str) -> List[str]:
        """Get appropriate authorized external scenarios for each agent type"""
        scenarios = {
            "legal": [
                "NDA review with external counsel at established law firm",
                "Legal matter coordination with co-counsel",
                "Contract negotiation with client's legal team",
                "Regulatory compliance discussion with government agency"
            ],
            
            "finance": [
                "Payment processing with authorized bank partner",
                "Invoice submission to established vendor portal", 
                "Treasury operations with banking relationship",
                "Audit coordination with accounting firm"
            ],
            
            "hr": [
                "Benefits enrollment with insurance provider",
                "Background check coordination with screening company",
                "Payroll processing with established payroll vendor",
                "Training coordination with authorized learning platform"
            ],
            
            "security": [
                "Vulnerability disclosure to security vendor",
                "Compliance audit with authorized security firm",
                "Technical integration with approved security tools"
            ],
            
            "casual": [
                "Project coordination with established contractor",
                "Technical support with vendor support team"
            ]
        }
        
        return scenarios.get(agent_type, [])


# Example usage and testing
def test_business_context():
    """Test business context analysis"""
    analyzer = BusinessContextAnalyzer()
    
    test_cases = [
        {
            "description": "Legitimate legal sharing",
            "spans": [{"type": "NDA", "start": 10, "end": 13}],
            "user_role": "LEGAL",
            "recipients": ["counsel@smith-legal.com"],
            "content": "Please review the NDA agreement for client matter",
            "expected_exposure": 0,
            "expected_context": 1
        },
        {
            "description": "Legal doc to personal email (violation)",
            "spans": [{"type": "NDA", "start": 10, "end": 13}],
            "user_role": "LEGAL", 
            "recipients": ["personal@gmail.com"],
            "content": "Please review the NDA agreement for client matter",
            "expected_exposure": 1,
            "expected_context": 0
        },
        {
            "description": "Financial data to bank (authorized)",
            "spans": [{"type": "PAN", "start": 20, "end": 36}],
            "user_role": "FINANCE",
            "recipients": ["processing@chase.com"],
            "content": "Please process payment with card 4532-1234-5678-9012",
            "expected_exposure": 0,
            "expected_context": 1
        },
        {
            "description": "HR data to wrong role (violation)",
            "spans": [{"type": "SSN", "start": 15, "end": 26}],
            "user_role": "MARKETING",  # Wrong role
            "recipients": ["vendor@benefits.com"],
            "content": "Employee SSN 123-45-6789 for benefits",
            "expected_exposure": 1,
            "expected_context": 0
        }
    ]
    
    print("=== Business Context Analysis Tests ===\n")
    
    for test in test_cases:
        result = analyzer.analyze_sharing_context(
            test["spans"],
            test["user_role"], 
            test["recipients"],
            test["content"]
        )
        
        exposure_pass = result["exposure"] == test["expected_exposure"]
        context_pass = result["context"] == test["expected_context"]
        
        status = "✅ PASS" if (exposure_pass and context_pass) else "❌ FAIL"
        print(f"{status} {test['description']}")
        print(f"   Expected: exposure={test['expected_exposure']}, context={test['expected_context']}")
        print(f"   Actual:   exposure={result['exposure']}, context={result['context']}")
        print(f"   Analysis: {result['analysis']}")
        print()


if __name__ == "__main__":
    test_business_context()