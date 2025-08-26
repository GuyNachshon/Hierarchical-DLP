#!/usr/bin/env python3
"""
Semantic Obfuscation Technique Library for DLP Training

This module provides comprehensive semantic obfuscation techniques that go beyond
simple regex patterns to create sophisticated hiding methods that require 
semantic understanding to detect.

Focus: Generate data that looks normal but contains hidden sensitive information
that only semantic analysis (not regex) can catch.
"""

import re
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ObfuscationType(Enum):
    """Types of semantic obfuscation techniques"""
    NUMERICAL_WORDS = "numerical_words"          # "five three two six"
    TEXTUAL_DECOMPOSITION = "textual_decomp"     # "john dot smith at company dot com"
    SPATIAL_BREAKDOWN = "spatial_breakdown"      # "5 3 2 6   1 2 3 4"
    EUPHEMISMS = "euphemisms"                   # "sixteen-digit identifier"
    MULTILINGUAL = "multilingual"               # "cinco tres dos seis"
    CONTEXTUAL_HIDING = "contextual_hiding"     # "client payment method"
    PROGRESSIVE_REVEAL = "progressive_reveal"    # Info spread across conversation
    TECHNICAL_ENCODING = "technical_encoding"   # References to encoded data
    VERBAL_REFERENCE = "verbal_reference"       # "as discussed verbally"
    ATTACHMENT_HIDING = "attachment_hiding"     # "see secure document"


@dataclass
class ObfuscationTechnique:
    """Definition of a specific obfuscation technique"""
    name: str
    technique_type: ObfuscationType
    description: str
    example_input: str
    example_output: str
    sophistication_level: str  # "low", "medium", "high"
    business_context: str
    detection_difficulty: str  # "easy", "medium", "hard"


class SemanticObfuscationLibrary:
    """Library of semantic obfuscation techniques for DLP training"""
    
    def __init__(self):
        self.techniques = self._initialize_techniques()
        self.number_words = {
            0: ["zero", "oh", "null"],
            1: ["one", "first", "single"],
            2: ["two", "second", "double", "pair"],
            3: ["three", "third", "triple"],
            4: ["four", "fourth", "quad"],
            5: ["five", "fifth", "penta"],
            6: ["six", "sixth", "hexa"],
            7: ["seven", "seventh", "sept"],
            8: ["eight", "eighth", "octo"],
            9: ["nine", "ninth", "nona"]
        }
        self.multilingual_numbers = {
            "spanish": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
            "french": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
            "german": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
        }
    
    def _initialize_techniques(self) -> List[ObfuscationTechnique]:
        """Initialize comprehensive obfuscation technique library"""
        return [
            # Numerical Word Techniques
            ObfuscationTechnique(
                name="credit_card_words",
                technique_type=ObfuscationType.NUMERICAL_WORDS,
                description="Convert credit card digits to words",
                example_input="4532-1234-5678-9012",
                example_output="four five three two one two three four five six seven eight nine zero one two",
                sophistication_level="medium",
                business_context="Payment processing, financial communications",
                detection_difficulty="medium"
            ),
            
            ObfuscationTechnique(
                name="ssn_verbal",
                technique_type=ObfuscationType.NUMERICAL_WORDS,
                description="SSN as spoken words",
                example_input="123-45-6789",
                example_output="one two three dash four five dash six seven eight nine",
                sophistication_level="medium", 
                business_context="HR, employee data, benefits",
                detection_difficulty="medium"
            ),
            
            # Textual Decomposition
            ObfuscationTechnique(
                name="email_decomposition",
                technique_type=ObfuscationType.TEXTUAL_DECOMPOSITION,
                description="Break email into components",
                example_input="john.smith@company.com",
                example_output="john dot smith at company dot com",
                sophistication_level="low",
                business_context="Any communication referencing emails",
                detection_difficulty="easy"
            ),
            
            ObfuscationTechnique(
                name="url_decomposition", 
                technique_type=ObfuscationType.TEXTUAL_DECOMPOSITION,
                description="Break URLs into components",
                example_input="https://secure.bank.com/account/12345",
                example_output="h t t p s colon slash slash secure dot bank dot com slash account slash one two three four five",
                sophistication_level="medium",
                business_context="IT, security, system access",
                detection_difficulty="medium"
            ),
            
            # Spatial Breakdown
            ObfuscationTechnique(
                name="spaced_numbers",
                technique_type=ObfuscationType.SPATIAL_BREAKDOWN,
                description="Insert strategic spacing in numbers",
                example_input="4532123456789012",
                example_output="4 5 3 2   1 2 3 4   5 6 7 8   9 0 1 2",
                sophistication_level="low",
                business_context="Casual typing, formatting issues",
                detection_difficulty="easy"
            ),
            
            # Euphemisms
            ObfuscationTechnique(
                name="payment_euphemisms",
                technique_type=ObfuscationType.EUPHEMISMS,
                description="Use business euphemisms for sensitive data",
                example_input="Credit card number",
                example_output="sixteen-digit payment identifier",
                sophistication_level="high",
                business_context="Professional, formal communications",
                detection_difficulty="hard"
            ),
            
            ObfuscationTechnique(
                name="credential_euphemisms",
                technique_type=ObfuscationType.EUPHEMISMS,
                description="Professional terms for credentials",
                example_input="Password: abc123",
                example_output="authentication credential: alpha-bravo-charlie-one-two-three",
                sophistication_level="high",
                business_context="IT, security, system access",
                detection_difficulty="hard"
            ),
            
            # Multilingual
            ObfuscationTechnique(
                name="spanish_numbers",
                technique_type=ObfuscationType.MULTILINGUAL,
                description="Use Spanish number words",
                example_input="4532",
                example_output="cuatro cinco tres dos",
                sophistication_level="medium",
                business_context="Multilingual workplace, international business",
                detection_difficulty="medium"
            ),
            
            # Contextual Hiding
            ObfuscationTechnique(
                name="business_context_hiding",
                technique_type=ObfuscationType.CONTEXTUAL_HIDING,
                description="Hide sensitive data in business context",
                example_input="API key: sk-abc123def456",
                example_output="the client service key begins with sierra-kilo and includes alpha-bravo-charlie-one-two-three",
                sophistication_level="high",
                business_context="Professional technical discussions",
                detection_difficulty="hard"
            ),
            
            # Technical Encoding References
            ObfuscationTechnique(
                name="encoding_references",
                technique_type=ObfuscationType.TECHNICAL_ENCODING,
                description="Reference to encoded/hidden data",
                example_input="Password: secretpass123",
                example_output="base64 encoded auth string in yesterday's email",
                sophistication_level="high",
                business_context="Technical communications, DevOps",
                detection_difficulty="hard"
            ),
            
            # Attachment/External Hiding
            ObfuscationTechnique(
                name="external_reference",
                technique_type=ObfuscationType.ATTACHMENT_HIDING,
                description="Reference sensitive data in external location",
                example_input="Credit card: 4532-1234-5678-9012",
                example_output="payment details in the secure folder, filename client_payment_q4.txt",
                sophistication_level="high",
                business_context="Document management, file sharing",
                detection_difficulty="hard"
            )
        ]
    
    def get_techniques_by_sophistication(self, level: str) -> List[ObfuscationTechnique]:
        """Get techniques by sophistication level"""
        return [t for t in self.techniques if t.sophistication_level == level]
    
    def get_techniques_by_type(self, obfuscation_type: ObfuscationType) -> List[ObfuscationTechnique]:
        """Get techniques by type"""
        return [t for t in self.techniques if t.technique_type == obfuscation_type]
    
    def convert_number_to_words(self, number_str: str, style: str = "english") -> str:
        """Convert numeric string to words"""
        if style == "english":
            words = []
            for digit in number_str:
                if digit.isdigit():
                    words.append(random.choice(self.number_words[int(digit)]))
                elif digit in "-.":
                    words.append("dash" if digit == "-" else "dot")
                else:
                    words.append(digit)
            return " ".join(words)
        
        elif style in self.multilingual_numbers:
            words = []
            for digit in number_str:
                if digit.isdigit():
                    words.append(self.multilingual_numbers[style][int(digit)])
                elif digit in "-.":
                    words.append("guión" if digit == "-" else "punto")  # Spanish example
                else:
                    words.append(digit)
            return " ".join(words)
        
        return number_str
    
    def decompose_email(self, email: str, style: str = "basic") -> str:
        """Convert email to decomposed form"""
        if style == "basic":
            return email.replace(".", " dot ").replace("@", " at ")
        elif style == "verbose":
            return email.replace(".", " point ").replace("@", " at the domain ")
        elif style == "spelled":
            result = ""
            for char in email:
                if char == ".":
                    result += " dot "
                elif char == "@":
                    result += " at "
                elif char.isalpha():
                    result += char + " "
                else:
                    result += char + " "
            return result.strip()
        
        return email
    
    def add_strategic_spacing(self, text: str, spacing_pattern: str = "groups") -> str:
        """Add strategic spacing to break patterns"""
        if spacing_pattern == "groups":
            # Add spaces in groups of 4
            result = ""
            for i, char in enumerate(text):
                if i > 0 and i % 4 == 0:
                    result += "   "  # Triple space for group separation
                elif i > 0 and i % 2 == 0:
                    result += " "    # Single space for pair separation
                result += char
            return result
        
        elif spacing_pattern == "random":
            # Add random spacing
            result = ""
            for i, char in enumerate(text):
                if i > 0 and random.random() < 0.3:
                    result += " "
                result += char
            return result
        
        return text
    
    def apply_euphemism(self, sensitive_type: str, context: str = "business") -> str:
        """Generate euphemisms for sensitive data types"""
        euphemisms = {
            "credit_card": {
                "business": ["payment instrument", "sixteen-digit identifier", "client payment method", "card credentials"],
                "casual": ["the payment thing", "card info", "payment details"],
                "technical": ["PCI data", "payment token", "card PAN"]
            },
            "ssn": {
                "business": ["social identifier", "tax identification", "personal reference number"],
                "casual": ["social security thing", "social number", "personal ID"],
                "technical": ["SSI data", "tax ID", "government identifier"]
            },
            "password": {
                "business": ["authentication credential", "access token", "security key"],
                "casual": ["login info", "access code", "the password"],
                "technical": ["auth string", "credential hash", "access token"]
            },
            "api_key": {
                "business": ["service credential", "integration key", "access token"],
                "casual": ["API thing", "service key", "access code"],
                "technical": ["bearer token", "service auth", "client credential"]
            }
        }
        
        if sensitive_type in euphemisms and context in euphemisms[sensitive_type]:
            return random.choice(euphemisms[sensitive_type][context])
        
        return f"{sensitive_type} information"
    
    def create_progressive_reveal_sequence(self, sensitive_data: str, num_parts: int = 3) -> List[str]:
        """Split sensitive data across multiple messages"""
        if sensitive_data.isdigit() and len(sensitive_data) >= num_parts:
            # Split numeric data
            chunk_size = len(sensitive_data) // num_parts
            parts = []
            for i in range(0, len(sensitive_data), chunk_size):
                chunk = sensitive_data[i:i+chunk_size]
                parts.append(chunk)
            
            # Create natural business contexts
            contexts = [
                f"The first part is {parts[0]}",
                f"Middle section: {parts[1] if len(parts) > 1 else ''}",
                f"Final digits are {parts[-1]}"
            ]
            return contexts[:len(parts)]
        
        # For non-numeric, split by words or characters
        words = sensitive_data.split()
        if len(words) >= num_parts:
            chunk_size = len(words) // num_parts
            parts = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                parts.append(chunk)
            return [f"Part {i+1}: {part}" for i, part in enumerate(parts)]
        
        return [sensitive_data]  # Can't split meaningfully
    
    def generate_obfuscated_example(self, sensitive_data: str, data_type: str, 
                                   sophistication: str = "medium", 
                                   business_context: str = "general") -> Dict[str, Any]:
        """Generate a complete obfuscated example"""
        
        # Select appropriate techniques
        techniques = self.get_techniques_by_sophistication(sophistication)
        applicable_techniques = [t for t in techniques if data_type.lower() in t.business_context.lower()]
        
        if not applicable_techniques:
            applicable_techniques = techniques
        
        chosen_technique = random.choice(applicable_techniques)
        
        # Apply the obfuscation
        obfuscated_data = self._apply_technique(sensitive_data, data_type, chosen_technique)
        
        # Generate business context
        business_reason = self._generate_business_context(data_type, business_context, sophistication)
        
        return {
            "original_data": sensitive_data,
            "obfuscated_data": obfuscated_data,
            "technique_used": chosen_technique.name,
            "technique_type": chosen_technique.technique_type.value,
            "sophistication_level": sophistication,
            "business_context": business_reason,
            "detection_difficulty": chosen_technique.detection_difficulty,
            "full_sentence": self._create_natural_sentence(obfuscated_data, business_reason)
        }
    
    def _apply_technique(self, data: str, data_type: str, technique: ObfuscationTechnique) -> str:
        """Apply specific obfuscation technique to data"""
        
        if technique.technique_type == ObfuscationType.NUMERICAL_WORDS:
            # Convert numbers to words
            if data_type == "credit_card":
                clean_data = re.sub(r'[^0-9]', '', data)
                return self.convert_number_to_words(clean_data)
            elif data_type == "ssn":
                return self.convert_number_to_words(data)
            elif data_type == "phone":
                clean_data = re.sub(r'[^0-9]', '', data)
                return self.convert_number_to_words(clean_data)
        
        elif technique.technique_type == ObfuscationType.TEXTUAL_DECOMPOSITION:
            if data_type == "email":
                return self.decompose_email(data, "basic")
            elif data_type == "url":
                return data.replace(".", " dot ").replace("://", " colon slash slash ").replace("/", " slash ")
        
        elif technique.technique_type == ObfuscationType.SPATIAL_BREAKDOWN:
            return self.add_strategic_spacing(data, "groups")
        
        elif technique.technique_type == ObfuscationType.EUPHEMISMS:
            euphemism = self.apply_euphemism(data_type, "business")
            if data.isdigit():
                return f"{euphemism}: {self.convert_number_to_words(data)}"
            else:
                return f"{euphemism} details"
        
        elif technique.technique_type == ObfuscationType.MULTILINGUAL:
            if data.isdigit():
                return self.convert_number_to_words(data, "spanish")
        
        elif technique.technique_type == ObfuscationType.CONTEXTUAL_HIDING:
            return f"the {data_type} mentioned in our call"
        
        elif technique.technique_type == ObfuscationType.TECHNICAL_ENCODING:
            return f"base64 encoded {data_type} in secure storage"
        
        elif technique.technique_type == ObfuscationType.ATTACHMENT_HIDING:
            return f"{data_type} details in attached secure document"
        
        # Fallback
        return data
    
    def _generate_business_context(self, data_type: str, context: str, sophistication: str) -> str:
        """Generate realistic business context for obfuscation"""
        contexts = {
            "credit_card": {
                "low": "payment processing setup",
                "medium": "client billing coordination", 
                "high": "confidential payment method verification"
            },
            "ssn": {
                "low": "employee record update",
                "medium": "benefits enrollment coordination",
                "high": "confidential identity verification process"
            },
            "password": {
                "low": "system access setup",
                "medium": "security credential rotation",
                "high": "confidential system authentication"
            },
            "api_key": {
                "low": "service integration",
                "medium": "production deployment coordination",
                "high": "confidential system authentication"
            }
        }
        
        if data_type in contexts and sophistication in contexts[data_type]:
            return contexts[data_type][sophistication]
        
        return f"{data_type} handling for business purposes"
    
    def _create_natural_sentence(self, obfuscated_data: str, business_context: str) -> str:
        """Create natural business sentence with obfuscated data"""
        sentence_templates = [
            f"For the {business_context}, we need {obfuscated_data}.",
            f"Regarding {business_context}, the details are {obfuscated_data}.",
            f"The {business_context} requires {obfuscated_data} for completion.",
            f"Per our discussion about {business_context}, use {obfuscated_data}.",
            f"To proceed with {business_context}, please confirm {obfuscated_data}."
        ]
        
        return random.choice(sentence_templates)


# Example usage and testing
def demo_semantic_obfuscation():
    """Demonstrate semantic obfuscation techniques"""
    library = SemanticObfuscationLibrary()
    
    # Test data
    test_cases = [
        ("4532-1234-5678-9012", "credit_card"),
        ("123-45-6789", "ssn"),
        ("john.smith@company.com", "email"),
        ("sk-abc123def456ghi789", "api_key"),
        ("SecretPass123!", "password")
    ]
    
    print("=== Semantic Obfuscation Demo ===\n")
    
    for data, data_type in test_cases:
        print(f"Original {data_type}: {data}")
        
        for sophistication in ["low", "medium", "high"]:
            result = library.generate_obfuscated_example(data, data_type, sophistication)
            print(f"  {sophistication.upper()}: {result['full_sentence']}")
            print(f"    Technique: {result['technique_used']} ({result['detection_difficulty']} to detect)")
        
        print()


if __name__ == "__main__":
    demo_semantic_obfuscation()