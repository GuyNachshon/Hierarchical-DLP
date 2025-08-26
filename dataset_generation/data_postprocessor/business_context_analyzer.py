"""
Business Context Analyzer for DLP Training Data Post-Processing

Analyzes business context to determine exposure risk and legitimate workflow indicators.
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class ContextAnalysis:
    """Results of business context analysis"""
    exposure: int  # 0 = low risk, 1 = high risk
    context: int   # 0 = suspicious/illegitimate, 1 = legitimate workflow
    risk_factors: List[str]
    legitimate_indicators: List[str]
    recipient_analysis: Dict[str, str]


class BusinessContextAnalyzer:
    """Analyzes business context for exposure and legitimacy assessment"""
    
    def __init__(self):
        """Initialize with domain knowledge and patterns"""
        self._setup_domain_knowledge()
        self._setup_patterns()
    
    def _setup_domain_knowledge(self):
        """Setup domain classification knowledge"""
        
        # High-risk personal domains
        self.personal_domains = {
            'gmail.com', 'outlook.com', 'hotmail.com', 'yahoo.com', 
            'proton.me', 'protonmail.com', 'icloud.com', 'me.com',
            'aol.com', 'mail.com', 'yandex.com', 'zoho.com'
        }
        
        # Legitimate business/organizational domains
        self.business_domain_patterns = [
            r'\.gov$',  # Government
            r'\.edu$',  # Educational
            r'\.org$',  # Organizations
            r'law\..*', # Law firms
            r'legal\..*', # Legal services
            r'accounting\..*', # Accounting firms
            r'consulting\..*'  # Consulting
        ]
        
        # High-risk external domains (file sharing, temp email, etc.)
        self.high_risk_domains = {
            'dropbox.com', 'drive.google.com', 'onedrive.com', 
            'sharepoint.com', 'box.com', 'wetransfer.com',
            'temp-mail.org', '10minutemail.com', 'guerrillamail.com',
            'mailinator.com', 'tempmail.io'
        }
        
        # Legitimate business communication indicators
        self.legitimate_patterns = {
            'legal': [
                r'\b(?:nda|confidentiality|non-disclosure|agreement|contract)\b',
                r'\b(?:counsel|attorney|legal|law firm|matter)\b',
                r'\b(?:privileged|confidential|attorney-client)\b',
                r'\b(?:engagement|retainer|legal services)\b'
            ],
            'finance': [
                r'\b(?:audit|financial|accounting|tax|compliance)\b',
                r'\b(?:invoice|billing|payment|accounting)\b', 
                r'\b(?:quarterly|annual|financial report)\b',
                r'\b(?:sec filing|10-k|10-q|proxy)\b'
            ],
            'hr': [
                r'\b(?:employment|hr|human resources|personnel)\b',
                r'\b(?:benefits|compensation|performance|review)\b',
                r'\b(?:onboarding|training|employee handbook)\b',
                r'\b(?:policy|procedure|compliance training)\b'
            ],
            'business_ops': [
                r'\b(?:vendor|supplier|partnership|collaboration)\b',
                r'\b(?:project|timeline|deliverables|milestone)\b',
                r'\b(?:meeting|conference|presentation|proposal)\b',
                r'\b(?:strategy|planning|roadmap|objectives)\b'
            ]
        }
        
        # Red flag patterns that indicate suspicious activity
        self.suspicious_patterns = [
            r'\b(?:urgent|asap|immediate|emergency)\b.*(?:send|share|provide)',
            r'\b(?:personal|private|secret|confidential)\b.*(?:information|data)',
            r'\b(?:password|login|access|credentials)\b',
            r'\b(?:delete|destroy|remove|erase)\b.*(?:after|once)',
            r'\b(?:don\'t tell|keep quiet|between us|confidential)\b',
            r'\b(?:wire transfer|bitcoin|crypto|untraceable)\b'
        ]
    
    def _setup_patterns(self):
        """Compile regex patterns for efficient matching"""
        
        # Compile legitimate patterns
        self.compiled_legitimate = {}
        for category, patterns in self.legitimate_patterns.items():
            self.compiled_legitimate[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns
            ]
        
        # Compile suspicious patterns  
        self.compiled_suspicious = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.suspicious_patterns
        ]
        
        # Business domain patterns
        self.compiled_business_domains = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.business_domain_patterns
        ]
    
    def analyze_sharing_context(
        self, 
        spans: List[Dict], 
        user_role: str, 
        recipients: List[str], 
        body: str, 
        sender_domain: Optional[str] = None
    ) -> ContextAnalysis:
        """
        Analyze business sharing context to determine exposure risk and legitimacy
        
        Args:
            spans: Extracted PII spans
            user_role: Role of sender (LEGAL, HR, FINANCE, etc.)
            recipients: List of recipient email addresses
            body: Email/message body content
            sender_domain: Sender's organization domain
            
        Returns:
            ContextAnalysis with exposure/context labels and analysis details
        """
        
        # Analyze recipients
        recipient_analysis = self._analyze_recipients(recipients, sender_domain)
        
        # Analyze content for legitimate vs suspicious indicators
        content_analysis = self._analyze_content(body, user_role, spans)
        
        # Calculate exposure risk
        exposure = self._calculate_exposure_risk(
            recipient_analysis, spans, content_analysis
        )
        
        # Calculate context legitimacy
        context = self._calculate_context_legitimacy(
            content_analysis, user_role, recipient_analysis, spans
        )
        
        return ContextAnalysis(
            exposure=exposure,
            context=context,
            risk_factors=content_analysis['risk_factors'] + recipient_analysis['risk_factors'],
            legitimate_indicators=content_analysis['legitimate_indicators'],
            recipient_analysis=recipient_analysis['categorization']
        )
    
    def _analyze_recipients(
        self, 
        recipients: List[str], 
        sender_domain: Optional[str]
    ) -> Dict:
        """Analyze recipient risk profile"""
        
        analysis = {
            'categorization': {},
            'risk_factors': [],
            'internal_count': 0,
            'external_business_count': 0,
            'external_personal_count': 0,
            'high_risk_count': 0
        }
        
        for recipient in recipients:
            if '@' not in recipient:
                continue
                
            domain = recipient.split('@')[-1].lower()
            
            # Categorize recipient
            if sender_domain and domain == sender_domain.lower():
                category = 'internal'
                analysis['internal_count'] += 1
            elif domain in self.personal_domains:
                category = 'external_personal'
                analysis['external_personal_count'] += 1
                analysis['risk_factors'].append(f"Personal email domain: {domain}")
            elif domain in self.high_risk_domains:
                category = 'high_risk'
                analysis['high_risk_count'] += 1
                analysis['risk_factors'].append(f"High-risk domain: {domain}")
            elif any(pattern.match(domain) for pattern in self.compiled_business_domains):
                category = 'external_business'
                analysis['external_business_count'] += 1
            else:
                # Unknown external domain - treat as business but flag for review
                category = 'external_business'
                analysis['external_business_count'] += 1
            
            analysis['categorization'][recipient] = category
        
        return analysis
    
    def _analyze_content(
        self, 
        body: str, 
        user_role: str, 
        spans: List[Dict]
    ) -> Dict:
        """Analyze content for legitimate vs suspicious indicators"""
        
        analysis = {
            'legitimate_indicators': [],
            'risk_factors': [],
            'legitimate_categories': set(),
            'suspicious_score': 0
        }
        
        # Check for legitimate business patterns
        for category, patterns in self.compiled_legitimate.items():
            for pattern in patterns:
                matches = pattern.findall(body)
                if matches:
                    analysis['legitimate_categories'].add(category)
                    analysis['legitimate_indicators'].extend(matches)
        
        # Check for suspicious patterns
        for pattern in self.compiled_suspicious:
            matches = pattern.findall(body)
            if matches:
                analysis['suspicious_score'] += len(matches)
                analysis['risk_factors'].extend([f"Suspicious pattern: {match}" for match in matches])
        
        # Role-specific legitimacy analysis
        role_legitimacy = self._analyze_role_specific_legitimacy(user_role, body, spans)
        analysis['legitimate_indicators'].extend(role_legitimacy)
        
        return analysis
    
    def _analyze_role_specific_legitimacy(
        self, 
        user_role: str, 
        body: str, 
        spans: List[Dict]
    ) -> List[str]:
        """Analyze legitimacy based on user role and context"""
        
        indicators = []
        role = user_role.upper() if user_role else ""
        
        # Legal role legitimacy
        if role == "LEGAL":
            if any(span.get('type') in ['NDA', 'MATTER', 'CASE'] for span in spans):
                indicators.append("Legal role with appropriate document references")
            if re.search(r'\b(?:privilege|confidential|attorney-client)\b', body, re.IGNORECASE):
                indicators.append("Legal privilege language")
        
        # HR role legitimacy  
        elif role == "HR":
            if re.search(r'\b(?:employee|benefits|performance|training)\b', body, re.IGNORECASE):
                indicators.append("HR role with appropriate content")
            if re.search(r'\b(?:confidential|personnel|employment)\b', body, re.IGNORECASE):
                indicators.append("HR confidentiality context")
        
        # Finance role legitimacy
        elif role == "FINANCE":
            if re.search(r'\b(?:audit|financial|accounting|compliance)\b', body, re.IGNORECASE):
                indicators.append("Finance role with appropriate content")
            if any(span.get('type') == 'PAN' for span in spans):
                indicators.append("Finance role handling payment data")
        
        # Security role legitimacy
        elif role == "SECURITY":
            if re.search(r'\b(?:incident|security|breach|vulnerability)\b', body, re.IGNORECASE):
                indicators.append("Security role with appropriate content")
        
        return indicators
    
    def _calculate_exposure_risk(
        self, 
        recipient_analysis: Dict, 
        spans: List[Dict], 
        content_analysis: Dict
    ) -> int:
        """Calculate exposure risk (0 = low, 1 = high)"""
        
        risk_score = 0
        
        # High risk: external personal domains
        if recipient_analysis['external_personal_count'] > 0:
            risk_score += 3
        
        # High risk: high-risk domains
        if recipient_analysis['high_risk_count'] > 0:
            risk_score += 4
        
        # Medium risk: external business domains
        if recipient_analysis['external_business_count'] > 0:
            risk_score += 1
        
        # Risk based on PII sensitivity
        sensitive_pii = sum(1 for span in spans if span.get('type') in ['PAN', 'SSN', 'SECRET', 'DBURI'])
        risk_score += sensitive_pii * 2
        
        # Risk based on suspicious content
        risk_score += content_analysis['suspicious_score']
        
        # Threshold: >= 3 is high risk
        return 1 if risk_score >= 3 else 0
    
    def _calculate_context_legitimacy(
        self, 
        content_analysis: Dict, 
        user_role: str, 
        recipient_analysis: Dict, 
        spans: List[Dict]
    ) -> int:
        """Calculate context legitimacy (0 = suspicious, 1 = legitimate)"""
        
        legitimacy_score = 0
        
        # Strong legitimate indicators
        if content_analysis['legitimate_indicators']:
            legitimacy_score += len(content_analysis['legitimate_indicators'])
        
        # Role-appropriate communication
        role_appropriate = (
            (user_role == "LEGAL" and 'legal' in content_analysis['legitimate_categories']) or
            (user_role == "HR" and 'hr' in content_analysis['legitimate_categories']) or
            (user_role == "FINANCE" and 'finance' in content_analysis['legitimate_categories']) or
            ('business_ops' in content_analysis['legitimate_categories'])
        )
        if role_appropriate:
            legitimacy_score += 2
        
        # Internal communications are generally more legitimate
        if recipient_analysis['internal_count'] > 0:
            legitimacy_score += 1
        
        # Business-to-business communication
        if recipient_analysis['external_business_count'] > 0 and recipient_analysis['external_personal_count'] == 0:
            legitimacy_score += 1
        
        # Penalize suspicious content
        legitimacy_score -= content_analysis['suspicious_score'] * 2
        
        # Penalize high-risk recipients
        legitimacy_score -= recipient_analysis['external_personal_count'] * 2
        legitimacy_score -= recipient_analysis['high_risk_count'] * 3
        
        # Threshold: >= 2 is legitimate
        return 1 if legitimacy_score >= 2 else 0
    
    def infer_sender_domain(self, recipients: List[str], body: str) -> Optional[str]:
        """
        Infer sender's organization domain from context
        
        Args:
            recipients: List of recipient emails
            body: Email body content
            
        Returns:
            Inferred organization domain or None
        """
        
        # Try to find internal domain from recipients
        domains = []
        for recipient in recipients:
            if '@' in recipient:
                domain = recipient.split('@')[-1].lower()
                if domain not in self.personal_domains and domain not in self.high_risk_domains:
                    domains.append(domain)
        
        # Find most common non-personal domain
        if domains:
            domain_counts = {}
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Return most frequent business domain
            return max(domain_counts.items(), key=lambda x: x[1])[0]
        
        # Try to extract from email signatures or body
        domain_pattern = re.compile(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
        matches = domain_pattern.findall(body)
        for domain in matches:
            domain = domain.lower()
            if domain not in self.personal_domains and domain not in self.high_risk_domains:
                return domain
        
        # Fallback
        return "company.com"


def analyze_business_domain_type(domain: str) -> str:
    """
    Classify domain type for business analysis
    
    Args:
        domain: Email domain to classify
        
    Returns:
        Domain type: 'personal', 'business', 'government', 'education', 'high_risk'
    """
    
    analyzer = BusinessContextAnalyzer()
    domain = domain.lower()
    
    if domain in analyzer.personal_domains:
        return 'personal'
    elif domain in analyzer.high_risk_domains:
        return 'high_risk'
    elif any(pattern.match(domain) for pattern in analyzer.compiled_business_domains):
        return 'business'
    elif domain.endswith('.gov'):
        return 'government'
    elif domain.endswith('.edu'):
        return 'education'
    else:
        return 'business'  # Default assumption for unknown domains