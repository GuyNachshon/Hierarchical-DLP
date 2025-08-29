#!/usr/bin/env python3
"""
Contextual DLP Labeling Strategy

Focuses on semantic understanding and business context rather than 
pattern matching (which regex handles better). This creates a model
that complements rather than competes with traditional DLP tools.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class ContextualDLPLabeler:
    """
    Advanced labeling that focuses on contextual understanding rather than pattern matching.
    
    Philosophy: Let regex handle patterns, let ML handle context and intent.
    """
    
    def __init__(self):
        # User role hierarchies and typical access patterns
        self.role_hierarchy = {
            'C-LEVEL': {'level': 5, 'access': 'all', 'trust': 0.9},
            'VP': {'level': 4, 'access': 'departmental+', 'trust': 0.8},
            'DIRECTOR': {'level': 4, 'access': 'departmental+', 'trust': 0.8},
            'MANAGER': {'level': 3, 'access': 'departmental', 'trust': 0.7},
            'SENIOR': {'level': 3, 'access': 'departmental', 'trust': 0.7},
            'LEAD': {'level': 3, 'access': 'team+', 'trust': 0.7},
            'EMPLOYEE': {'level': 2, 'access': 'team', 'trust': 0.6},
            'ANALYST': {'level': 2, 'access': 'team', 'trust': 0.6},
            'INTERN': {'level': 1, 'access': 'limited', 'trust': 0.3},
            'TEMP': {'level': 1, 'access': 'limited', 'trust': 0.2},
            'CONTRACTOR': {'level': 1, 'access': 'project', 'trust': 0.4},
        }
        
        # Department sensitivity levels
        self.dept_sensitivity = {
            'FINANCE': 0.9,
            'LEGAL': 0.9,
            'HR': 0.8,
            'EXECUTIVE': 0.9,
            'IT': 0.7,
            'SECURITY': 0.8,
            'COMPLIANCE': 0.8,
            'ENGINEERING': 0.6,
            'SALES': 0.5,
            'MARKETING': 0.4,
            'OPERATIONS': 0.5,
            'SUPPORT': 0.4,
        }
        
        # Business context patterns (not regex patterns!)
        self.business_contexts = {
            'legitimate_sharing': [
                'quarterly report', 'board meeting', 'audit preparation',
                'compliance review', 'legal discovery', 'merger due diligence',
                'performance review', 'budget planning', 'contract negotiation'
            ],
            'suspicious_contexts': [
                'keep this quiet', 'don\'t tell anyone', 'between you and me',
                'off the record', 'confidentially speaking', 'this stays here',
                'personal favor', 'side project', 'freelance work'
            ],
            'pressure_indicators': [
                'urgent', 'immediately', 'asap', 'rush', 'emergency',
                'deadline passed', 'overdue', 'critical', 'escalated'
            ],
            'financial_motivation': [
                'payment', 'compensation', 'bonus', 'incentive', 'reward',
                'commission', 'fee', 'consulting', 'contract work'
            ]
        }
    
    def analyze_user_context_appropriateness(self, user_info: Dict, content_type: str) -> float:
        """
        Analyze if the user's role/department is appropriate for the content type.
        This is what ML excels at - understanding business logic appropriateness.
        """
        role = user_info.get('role', 'EMPLOYEE').upper()
        dept = user_info.get('dept', 'GENERAL').upper()
        
        # Get user trust level and access rights
        role_info = self.role_hierarchy.get(role, {'level': 2, 'access': 'team', 'trust': 0.6})
        dept_sensitivity = self.dept_sensitivity.get(dept, 0.5)
        
        # Context-based risk assessment
        risk_score = 0.0
        
        # Low-level users with high-sensitivity content
        if role_info['trust'] < 0.4 and 'confidential' in content_type.lower():
            risk_score += 0.6
            
        # Cross-department sharing without business justification
        if 'financial' in content_type.lower() and dept not in ['FINANCE', 'LEGAL', 'EXECUTIVE']:
            risk_score += 0.4
            
        # Personal/contractor access to sensitive data
        if role in ['INTERN', 'TEMP', 'CONTRACTOR'] and dept_sensitivity > 0.7:
            risk_score += 0.5
            
        return min(risk_score, 1.0)
    
    def analyze_recipient_relationship_appropriateness(self, sender_info: Dict, recipients: List[str], content_summary: str) -> float:
        """
        Analyze if the sender-recipient relationship makes business sense.
        Focus on business logic, not domain patterns.
        """
        risk_score = 0.0
        
        sender_dept = sender_info.get('dept', 'GENERAL').upper()
        sender_role = sender_info.get('role', 'EMPLOYEE').upper()
        
        external_recipients = []
        internal_recipients = []
        
        # Classify recipients (basic domain check, but focus on relationship analysis)
        for recipient in recipients:
            if any(domain in recipient.lower() for domain in ['company.com', 'example.com']):
                internal_recipients.append(recipient)
            else:
                external_recipients.append(recipient)
        
        # Business relationship analysis
        if external_recipients:
            # High-sensitivity departments sharing externally
            if sender_dept in ['FINANCE', 'LEGAL', 'HR'] and len(external_recipients) > 0:
                risk_score += 0.4
                
            # Personal domains (business context inappropriate)
            personal_domains = ['gmail', 'yahoo', 'hotmail', 'personal']
            personal_count = sum(1 for r in external_recipients 
                               if any(d in r.lower() for d in personal_domains))
            if personal_count > 0:
                risk_score += 0.3 * personal_count
                
            # Volume analysis - many external recipients suspicious
            if len(external_recipients) > 3:
                risk_score += 0.3
        
        # Internal over-sharing (role-based)
        if sender_role in ['INTERN', 'TEMP'] and len(recipients) > 5:
            risk_score += 0.2
            
        return min(risk_score, 1.0)
    
    def analyze_business_context_legitimacy(self, subject: str, body: str) -> Tuple[float, float]:
        """
        Analyze the business context and intent indicators.
        Returns (context_risk, intent_risk)
        """
        text = (subject + " " + body).lower()
        
        context_risk = 0.0
        intent_risk = 0.0
        
        # Legitimate business context indicators
        legitimate_count = sum(1 for pattern in self.business_contexts['legitimate_sharing']
                             if pattern in text)
        if legitimate_count > 0:
            context_risk -= 0.2  # Reduce risk for legitimate contexts
            
        # Suspicious context indicators
        suspicious_count = sum(1 for pattern in self.business_contexts['suspicious_contexts']
                             if pattern in text)
        context_risk += suspicious_count * 0.3
        
        # Pressure/urgency indicators (social engineering)
        pressure_count = sum(1 for pattern in self.business_contexts['pressure_indicators']
                           if pattern in text)
        intent_risk += min(pressure_count * 0.2, 0.4)
        
        # Financial motivation indicators
        financial_count = sum(1 for pattern in self.business_contexts['financial_motivation']
                            if pattern in text)
        intent_risk += min(financial_count * 0.3, 0.6)
        
        # Language analysis for intent
        if any(phrase in text for phrase in ['selling', 'for sale', 'buyer', 'purchase']):
            intent_risk += 0.5
            
        if any(phrase in text for phrase in ['mistake', 'accidentally', 'wrong recipient']):
            context_risk += 0.3  # Accidental disclosure risk
            
        return min(context_risk, 1.0), min(intent_risk, 1.0)
    
    def analyze_data_sensitivity_context(self, content: str, attachments: List[Dict]) -> float:
        """
        Focus on data sensitivity based on business context, not pattern matching.
        """
        sensitivity_score = 0.0
        
        # Business sensitivity indicators (not regex patterns)
        business_sensitive_terms = [
            'merger', 'acquisition', 'due diligence', 'board decision',
            'layoffs', 'restructuring', 'budget cuts', 'strategy',
            'competitive analysis', 'market research', 'client list'
        ]
        
        content_lower = content.lower()
        business_sensitivity = sum(1 for term in business_sensitive_terms if term in content_lower)
        sensitivity_score += min(business_sensitivity * 0.2, 0.6)
        
        # Attachment context analysis
        for attachment in attachments:
            att_name = attachment.get('name', '').lower()
            att_size = attachment.get('size', 0)
            
            # Business document types (not file extension matching)
            if any(term in att_name for term in ['financial', 'budget', 'salary', 'contract', 'agreement']):
                sensitivity_score += 0.3
                
            # Unusual size patterns might indicate data dumps
            if att_size > 50000000:  # >50MB unusual for business docs
                sensitivity_score += 0.2
                
            # Multiple large attachments suggest bulk data
            if att_size > 10000000 and len(attachments) > 3:
                sensitivity_score += 0.3
                
        return min(sensitivity_score, 1.0)
    
    def generate_contextual_labels(self, example: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate labels focused on contextual understanding rather than pattern matching.
        
        New labeling philosophy:
        - Sensitivity: Business appropriateness and data context
        - Exposure: Relationship appropriateness and business justification  
        - Context: User permissions and business logic compliance
        - Obfuscation: Intent analysis and social engineering indicators
        """
        
        # Extract information
        user_info = example.get('user', {})
        subject = example.get('subject', '')
        body = example.get('body', '')
        recipients = example.get('recipients', [])
        attachments = example.get('attachments', [])
        
        # Contextual analysis
        user_context_risk = self.analyze_user_context_appropriateness(
            user_info, subject + " " + body
        )
        
        recipient_risk = self.analyze_recipient_relationship_appropriateness(
            user_info, recipients, subject + " " + body
        )
        
        business_context_risk, intent_risk = self.analyze_business_context_legitimacy(
            subject, body
        )
        
        data_sensitivity = self.analyze_data_sensitivity_context(
            subject + " " + body, attachments
        )
        
        # Combine into final labels
        # Focus on business context rather than pattern detection
        labels = {
            'sensitivity': data_sensitivity,  # Business data sensitivity context
            'exposure': recipient_risk,       # Relationship appropriateness  
            'context': user_context_risk + business_context_risk * 0.5,  # Business logic compliance
            'obfuscation': intent_risk        # Intent and social engineering analysis
        }
        
        # Ensure scores are in [0,1] range and add small realistic noise
        noise_factor = 0.02
        for key in labels:
            labels[key] = max(0.0, min(1.0, labels[key] + np.random.uniform(-noise_factor, noise_factor)))
        
        return labels

def compare_labeling_approaches():
    """Compare pattern-based vs contextual labeling on sample examples."""
    print("📊 Contextual vs Pattern-Based Labeling Comparison")
    print("=" * 60)
    
    # Sample examples to demonstrate the difference
    examples = [
        {
            "name": "Low-level user with financial data",
            "example": {
                "user": {"role": "INTERN", "dept": "IT"},
                "subject": "Q4 Financial Results Review", 
                "body": "Please review the quarterly financial summary for our team discussion.",
                "recipients": ["manager@company.com"],
                "attachments": [{"name": "Q4_financials.pdf", "size": 2048000}]
            },
            "pattern_focus": "Low risk - no SSNs/passwords detected",
            "contextual_focus": "Medium-High risk - Intern accessing financial data outside department"
        },
        {
            "name": "Senior exec with business justification",
            "example": {
                "user": {"role": "CFO", "dept": "FINANCE"}, 
                "subject": "Board Meeting Preparation",
                "body": "Preparing financial summary for board review. Employee compensation data attached for discussion.",
                "recipients": ["board-secretary@company.com"],
                "attachments": [{"name": "compensation_analysis.xlsx", "size": 1024000}]
            },
            "pattern_focus": "High risk - contains compensation data",
            "contextual_focus": "Low risk - Appropriate business context and user role"
        }
    ]
    
    contextual_labeler = ContextualDLPLabeler()
    
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: {ex['name']}")
        print("-" * 40)
        
        labels = contextual_labeler.generate_contextual_labels(ex['example'])
        
        print(f"📋 Scenario: {ex['example']['subject']}")
        print(f"👤 User: {ex['example']['user']['role']} from {ex['example']['user']['dept']}")
        print(f"📧 Recipients: {ex['example']['recipients']}")
        
        print(f"\n🤖 Pattern-based approach would say:")
        print(f"   {ex['pattern_focus']}")
        
        print(f"🧠 Contextual approach says:")
        print(f"   {ex['contextual_focus']}")
        print(f"   Labels: S:{labels['sensitivity']:.3f} E:{labels['exposure']:.3f} C:{labels['context']:.3f} O:{labels['obfuscation']:.3f}")

if __name__ == "__main__":
    compare_labeling_approaches()