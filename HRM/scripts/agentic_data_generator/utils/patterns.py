"""
Span detection patterns for traditional and semantic obfuscation techniques.
"""

import re
from typing import List, Dict


class SpanDetector:
    """Detects sensitive spans in content including semantically obfuscated ones."""

    def __init__(self):
        # Traditional regex patterns (for baseline comparison)
        self.traditional_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL"),
            (r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', "PHONE"),
            (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b', "PAN"),
            (r'\b\d{3}-?\d{2}-?\d{4}\b', "SSN"),
            (r'\bAKIA[A-Z0-9]{16}\b', "SECRET"),
            (r'\bsk-[A-Za-z0-9]{48}\b', "SECRET"),
            (r'\bghp_[A-Za-z0-9]{36}\b', "SECRET"),
            (r'\b(?:postgresql|mysql|mongodb)://[^\s]+\b', "DBURI"),
            (r'\b(?:NDA|Non-Disclosure|Matter [A-Z]-\d+)\b', "NDA"),
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', "NAME")
        ]

        # Semantic obfuscation patterns (require understanding, not just pattern matching)
        self.semantic_patterns = [
            # Number words for credit cards/SSNs
            (r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)){7,15}\b', "PAN_SEMANTIC"),
            (r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)){8}\b', "SSN_SEMANTIC"),
            
            # Email decomposition patterns
            (r'\b\w+\s+dot\s+\w+\s+at\s+\w+\s+dot\s+\w+\b', "EMAIL_SEMANTIC"),
            
            # Phone number words
            (r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)){9}\b', "PHONE_SEMANTIC"),
            
            # Spaced/broken numbers (credit card format)
            (r'\b\d\s+\d\s+\d\s+\d\s+\s+\d\s+\d\s+\d\s+\d\s+\s+\d\s+\d\s+\d\s+\d\s+\s+\d\s+\d\s+\d\s+\d\b', "PAN_SPATIAL"),
            
            # API key phonetic patterns
            (r'\bsierra[-\s]?kilo[-\s]?\w+', "SECRET_PHONETIC"),
            
            # Multilingual numbers (Spanish example)
            (r'\b(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)(?:\s+(?:cero|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)){3,15}\b', "MULTILINGUAL_SEMANTIC")
        ]

        # Euphemism detection patterns
        self.euphemism_patterns = [
            (r'\b(?:sixteen[-\s]?digit|payment\s+(?:identifier|method|instrument))\b', "PAN_EUPHEMISM"),
            (r'\b(?:authentication\s+credential|access\s+token|service\s+key)\b', "SECRET_EUPHEMISM"),
            (r'\b(?:social\s+identifier|tax\s+identification|personal\s+reference)\b', "SSN_EUPHEMISM"),
            (r'\b(?:confidentiality\s+agreement|non[-\s]?disclosure)\b', "NDA_EUPHEMISM"),
        ]

    def extract_all_spans(self, content: str) -> List[Dict]:
        """Extract all sensitive spans from content including semantically obfuscated ones."""
        spans = []

        # Apply all pattern types
        all_patterns = (
            self.traditional_patterns + 
            self.semantic_patterns + 
            self.euphemism_patterns
        )

        for pattern, span_type in all_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                spans.append({
                    "type": span_type,
                    "start": match.start(),
                    "end": match.end()
                })

        return spans

    def extract_traditional_spans(self, content: str) -> List[Dict]:
        """Extract only traditional pattern spans (for comparison)."""
        spans = []
        for pattern, span_type in self.traditional_patterns:
            for match in re.finditer(pattern, content):
                spans.append({
                    "type": span_type,
                    "start": match.start(),
                    "end": match.end()
                })
        return spans

    def extract_semantic_spans(self, content: str) -> List[Dict]:
        """Extract only semantic obfuscation spans."""
        spans = []
        for pattern, span_type in self.semantic_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                spans.append({
                    "type": span_type,
                    "start": match.start(),
                    "end": match.end()
                })
        return spans

    def identify_obfuscation_techniques(self, content: str) -> List[str]:
        """Identify which obfuscation techniques were used in the content."""
        techniques = []

        # Check for different technique types
        word_numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
        if any(word in content.lower() for word in word_numbers):
            techniques.append("numerical_words")

        if " dot " in content or " at " in content or " dash " in content:
            techniques.append("textual_decomposition")

        euphemism_words = ["identifier", "token", "credential", "authentication", "method"]
        if any(word in content.lower() for word in euphemism_words):
            techniques.append("euphemisms")

        reference_phrases = ["discussed", "mentioned", "attached", "secure", "base64"]
        if any(phrase in content.lower() for phrase in reference_phrases):
            techniques.append("contextual_reference")

        # Multilingual patterns
        spanish_numbers = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
        if any(word in content.lower() for word in spanish_numbers):
            techniques.append("multilingual")

        return techniques