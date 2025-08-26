"""
PII Extraction Module for DLP Training Data Post-Processing

Extracts PII spans from generated text with character-level positions.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class PIISpan:
    """Represents a PII span with type and position"""
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


class PIIExtractor:
    """Comprehensive PII extraction using regex patterns"""
    
    def __init__(self):
        """Initialize with PII regex patterns"""
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for different PII types"""
        
        # Email patterns (comprehensive)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone patterns (US/international)
        phone_patterns = [
            r'\b(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)?[2-9]\d{2}[-.\s]?\d{4}\b',  # US format
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Simple format
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # (555) 123-4567
            r'\+\d{1,3}[-.\s]?\d{6,14}\b'  # International
        ]
        self.phone_pattern = re.compile('|'.join(phone_patterns), re.IGNORECASE)
        
        # Credit Card (PAN) patterns
        pan_patterns = [
            r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Visa
            r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Mastercard
            r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b',  # Amex
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # Generic 16-digit
        ]
        self.pan_pattern = re.compile('|'.join(pan_patterns), re.IGNORECASE)
        
        # SSN patterns
        ssn_patterns = [
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',  # XXX-XX-XXXX
            r'\b\d{9}\b'  # XXXXXXXXX (9 digits)
        ]
        self.ssn_pattern = re.compile('|'.join(ssn_patterns))
        
        # Name patterns (sophisticated)
        # Captures common name patterns in business contexts
        self.name_pattern = re.compile(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)?\s+[A-Z][a-z]+\b',
            re.MULTILINE
        )
        
        # Address patterns
        address_patterns = [
            r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            r'\b[A-Z][a-z]+,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'  # City, ST ZIP
        ]
        self.address_pattern = re.compile('|'.join(address_patterns), re.IGNORECASE)
        
        # Database URI patterns
        db_uri_patterns = [
            r'(?:postgres|mysql|mongodb|redis)://[^\s]+',
            r'jdbc:[^\s]+',
            r'Server=[^;]+;Database=[^;]+',
            r'mongodb\+srv://[^\s]+'
        ]
        self.dburi_pattern = re.compile('|'.join(db_uri_patterns), re.IGNORECASE)
        
        # Secret/API Key patterns
        secret_patterns = [
            r'\b[A-Za-z0-9+/]{40,}={0,2}\b',  # Base64-like strings
            r'sk-[A-Za-z0-9]{40,}',  # OpenAI style
            r'ghp_[A-Za-z0-9]{36}',  # GitHub personal access token
            r'AIza[0-9A-Za-z_-]{35}',  # Google API key
            r'AKIA[0-9A-Z]{16}',  # AWS access key
            r'xoxb-[0-9]+-[0-9]+-[0-9A-Za-z]+',  # Slack bot token
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'  # UUID
        ]
        self.secret_pattern = re.compile('|'.join(secret_patterns), re.IGNORECASE)
        
        # Legal document patterns
        legal_patterns = [
            r'NDA\s*#?\d*',
            r'Matter\s*#?\d+',
            r'Case\s*#?\d+',
            r'Contract\s*#?\d+'
        ]
        self.legal_pattern = re.compile('|'.join(legal_patterns), re.IGNORECASE)
        
        # Semantic obfuscation patterns
        self.semantic_patterns = {
            'number_words': re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b', re.IGNORECASE),
            'decomposition': re.compile(r'\b(?:dot|at|dash|space|underscore)\b', re.IGNORECASE), 
            'euphemisms': re.compile(r'\b(?:identifier|credential|authentication|payment\s+method|access\s+key)\b', re.IGNORECASE),
            'references': re.compile(r'\b(?:discussed|mentioned|secure\s+folder|attached|as\s+agreed)\b', re.IGNORECASE)
        }
    
    def extract_all_pii(self, text: str) -> List[PIISpan]:
        """Extract all PII spans from text"""
        spans = []
        
        # Extract each PII type
        spans.extend(self._extract_emails(text))
        spans.extend(self._extract_phones(text))
        spans.extend(self._extract_pans(text))
        spans.extend(self._extract_ssns(text))
        spans.extend(self._extract_names(text))
        spans.extend(self._extract_addresses(text))
        spans.extend(self._extract_db_uris(text))
        spans.extend(self._extract_secrets(text))
        spans.extend(self._extract_legal_terms(text))
        
        # Remove overlapping spans (keep longer ones)
        spans = self._remove_overlaps(spans)
        
        # Sort by position
        spans.sort(key=lambda x: x.start)
        
        return spans
    
    def _extract_emails(self, text: str) -> List[PIISpan]:
        """Extract email addresses"""
        spans = []
        for match in self.email_pattern.finditer(text):
            spans.append(PIISpan(
                type="EMAIL",
                text=match.group(),
                start=match.start(),
                end=match.end()
            ))
        return spans
    
    def _extract_phones(self, text: str) -> List[PIISpan]:
        """Extract phone numbers"""
        spans = []
        for match in self.phone_pattern.finditer(text):
            # Validate phone number (must have enough digits)
            phone_digits = re.sub(r'[^\d]', '', match.group())
            if len(phone_digits) >= 7:  # Minimum valid phone
                spans.append(PIISpan(
                    type="PHONE",
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        return spans
    
    def _extract_pans(self, text: str) -> List[PIISpan]:
        """Extract credit card numbers (PANs)"""
        spans = []
        for match in self.pan_pattern.finditer(text):
            # Basic Luhn validation
            pan_digits = re.sub(r'[^\d]', '', match.group())
            if len(pan_digits) >= 13 and self._is_valid_luhn(pan_digits):
                spans.append(PIISpan(
                    type="PAN",
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        return spans
    
    def _extract_ssns(self, text: str) -> List[PIISpan]:
        """Extract Social Security Numbers"""
        spans = []
        for match in self.ssn_pattern.finditer(text):
            ssn_digits = re.sub(r'[^\d]', '', match.group())
            # Validate SSN format and avoid false positives
            if len(ssn_digits) == 9 and not ssn_digits.startswith('000'):
                spans.append(PIISpan(
                    type="SSN",
                    text=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        return spans
    
    def _extract_names(self, text: str) -> List[PIISpan]:
        """Extract person names"""
        spans = []
        for match in self.name_pattern.finditer(text):
            name = match.group()
            # Filter out common false positives
            if not self._is_common_false_positive_name(name):
                spans.append(PIISpan(
                    type="NAME",
                    text=name,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8  # Names are harder to validate
                ))
        return spans
    
    def _extract_addresses(self, text: str) -> List[PIISpan]:
        """Extract addresses"""
        spans = []
        for match in self.address_pattern.finditer(text):
            spans.append(PIISpan(
                type="ADDR",
                text=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
        return spans
    
    def _extract_db_uris(self, text: str) -> List[PIISpan]:
        """Extract database connection strings"""
        spans = []
        for match in self.dburi_pattern.finditer(text):
            spans.append(PIISpan(
                type="DBURI",
                text=match.group(),
                start=match.start(),
                end=match.end()
            ))
        return spans
    
    def _extract_secrets(self, text: str) -> List[PIISpan]:
        """Extract API keys and secrets"""
        spans = []
        for match in self.secret_pattern.finditer(text):
            secret = match.group()
            # Additional validation for secret patterns
            if len(secret) >= 16 and self._looks_like_secret(secret):
                spans.append(PIISpan(
                    type="SECRET",
                    text=secret,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        return spans
    
    def _extract_legal_terms(self, text: str) -> List[PIISpan]:
        """Extract legal document references"""
        spans = []
        for match in self.legal_pattern.finditer(text):
            legal_type = match.group().upper()
            if legal_type.startswith('NDA'):
                pii_type = "NDA"
            elif legal_type.startswith('MATTER'):
                pii_type = "MATTER" 
            elif legal_type.startswith('CASE'):
                pii_type = "CASE"
            else:
                pii_type = "LEGAL"
                
            spans.append(PIISpan(
                type=pii_type,
                text=match.group(),
                start=match.start(),
                end=match.end()
            ))
        return spans
    
    def detect_obfuscation_indicators(self, text: str) -> Dict[str, bool]:
        """Detect various obfuscation techniques"""
        indicators = {}
        
        # Traditional obfuscation
        text_lower = text.lower()
        indicators['base64'] = 'base64' in text_lower or 'encoded' in text_lower
        
        # Semantic obfuscation
        for technique, pattern in self.semantic_patterns.items():
            indicators[technique] = bool(pattern.search(text))
        
        # Multilingual obfuscation
        multilingual_numbers = re.compile(r'\b(?:uno|dos|tres|cuatro|cinco|sechs|sieben|acht)\b', re.IGNORECASE)
        indicators['multilingual'] = bool(multilingual_numbers.search(text))
        
        # Unicode obfuscation (zero-width chars, homoglyphs)
        indicators['unicode_tricks'] = self._has_unicode_obfuscation(text)
        
        return indicators
    
    def _is_valid_luhn(self, card_num: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        return luhn_checksum(card_num) == 0
    
    def _is_common_false_positive_name(self, name: str) -> bool:
        """Filter out common false positive names"""
        false_positives = {
            'Dear Sir', 'Thank You', 'Best Regards', 'Kind Regards',
            'John Doe', 'Jane Doe', 'Test User', 'Admin User',
            'First Last', 'Your Name', 'User Name'
        }
        return name in false_positives or len(name.split()) > 3
    
    def _looks_like_secret(self, text: str) -> bool:
        """Additional validation for secret-like strings"""
        # Must have mix of chars
        has_upper = any(c.isupper() for c in text)
        has_lower = any(c.islower() for c in text) 
        has_digit = any(c.isdigit() for c in text)
        
        return (has_upper or has_lower) and has_digit
    
    def _has_unicode_obfuscation(self, text: str) -> bool:
        """Detect unicode-based obfuscation techniques"""
        # Check for zero-width characters
        zero_width_chars = {'\u200B', '\u200C', '\u200D', '\uFEFF'}
        if any(char in text for char in zero_width_chars):
            return True
        
        # Check for suspicious unicode normalization differences
        nfc = unicodedata.normalize('NFC', text)
        nfd = unicodedata.normalize('NFD', text)
        if nfc != nfd and len(nfd) > len(nfc) * 1.1:
            return True
        
        return False
    
    def _remove_overlaps(self, spans: List[PIISpan]) -> List[PIISpan]:
        """Remove overlapping spans, keeping higher confidence/longer ones"""
        if not spans:
            return spans
        
        # Sort by start position
        spans.sort(key=lambda x: x.start)
        
        filtered = []
        for span in spans:
            # Check if this span overlaps with any existing span
            overlaps = False
            for existing in filtered:
                if (span.start < existing.end and span.end > existing.start):
                    # Overlapping - keep the better one
                    overlaps = True
                    if (span.confidence > existing.confidence or 
                        (span.confidence == existing.confidence and len(span.text) > len(existing.text))):
                        # Replace existing with current span
                        filtered.remove(existing)
                        filtered.append(span)
                    break
            
            if not overlaps:
                filtered.append(span)
        
        return filtered


def create_bio_tags_from_spans(spans: List[PIISpan], text_length: int) -> List[str]:
    """Convert PII spans to BIO tags for the entire text"""
    # Initialize with O (Outside) tags
    tags = ['O'] * text_length
    
    # Apply BIO tags for each span
    for span in spans:
        for i in range(span.start, min(span.end, text_length)):
            if i == span.start:
                tags[i] = f'B-{span.type}'
            else:
                tags[i] = f'I-{span.type}'
    
    return tags


# BIO tag vocabulary matching the training system
BIO_TAG_TO_ID = {
    'O': 0,
    'B-EMAIL': 1, 'I-EMAIL': 2,
    'B-PHONE': 3, 'I-PHONE': 4, 
    'B-PAN': 5, 'I-PAN': 6,
    'B-SSN': 7, 'I-SSN': 8,
    'B-NAME': 9, 'I-NAME': 10,
    'B-ADDR': 11, 'I-ADDR': 12,
    'B-SECRET': 13, 'I-SECRET': 14,
    'B-DBURI': 15, 'I-DBURI': 16,
    'B-NDA': 17, 'I-NDA': 18,
    'B-MATTER': 19, 'I-MATTER': 20
}

NUM_BIO_TAGS = len(BIO_TAG_TO_ID)