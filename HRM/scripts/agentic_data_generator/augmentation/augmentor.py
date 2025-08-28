"""
Dataset Augmentation Module for HRM-DLP Training Data

Creates realistic variations of base examples to expand dataset size and diversity.
Focuses on recipient patterns, content variations, and risk scenarios.
"""

import random
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy
import hashlib

@dataclass
class AugmentationConfig:
    """Configuration for dataset augmentation."""
    augmentation_ratio: float = 3.0  # How many variants per base example
    recipient_variation_prob: float = 0.7  # Probability of changing recipients
    content_variation_prob: float = 0.5   # Probability of content variations
    thread_variation_prob: float = 0.4    # Probability of thread variations
    obfuscation_prob: float = 0.3         # Probability of adding obfuscation
    
    # Recipient domain pools
    external_personal_domains: List[str] = None
    external_business_domains: List[str] = None
    internal_domains: List[str] = None
    
    def __post_init__(self):
        if self.external_personal_domains is None:
            self.external_personal_domains = [
                "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
                "proton.me", "aol.com", "icloud.com", "mail.com"
            ]
        if self.external_business_domains is None:
            self.external_business_domains = [
                "acmelegal.com", "smithlaw.com", "jonesconsulting.net",
                "techsolutions.io", "businesspartners.com", "vendorcorp.net",
                "commercialbank.com", "insurancepartner.com"
            ]
        if self.internal_domains is None:
            self.internal_domains = [
                "company.com", "corp.internal", "enterprise.local"
            ]


class DatasetAugmentor:
    """Creates realistic variations of base examples for dataset expansion."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.name_pool = self._generate_name_pool()
        self.email_prefixes = self._generate_email_prefixes()
        
    def _generate_name_pool(self) -> List[str]:
        """Generate pool of realistic names for variations."""
        first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "James", "Jennifer",
            "Robert", "Lisa", "William", "Karen", "Richard", "Susan", "Charles", "Betty",
            "Thomas", "Helen", "Christopher", "Sandra", "Daniel", "Donna", "Matthew", "Carol"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
        ]
        return [f"{first} {last}" for first in first_names for last in last_names[:8]]
    
    def _generate_email_prefixes(self) -> List[str]:
        """Generate realistic email prefixes."""
        return [
            "john.doe", "j.smith", "sarah.wilson", "m.johnson", "david.brown",
            "admin", "support", "manager", "director", "counsel", "finance",
            "hr", "legal", "security", "accounting", "operations"
        ]
    
    def augment_example(self, base_example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create multiple variations of a base example."""
        variants = []
        num_variants = int(self.config.augmentation_ratio)
        
        for i in range(num_variants):
            variant = deepcopy(base_example)
            
            # Apply different types of variations with validation
            if random.random() < self.config.recipient_variation_prob:
                variant = self._vary_recipients(variant)
                if not isinstance(variant, dict):
                    print(f"⚠️  _vary_recipients returned {type(variant)} instead of dict")
                    continue
            
            if random.random() < self.config.content_variation_prob:
                variant = self._vary_content(variant)
                if not isinstance(variant, dict):
                    print(f"⚠️  _vary_content returned {type(variant)} instead of dict")
                    continue
                
            if random.random() < self.config.thread_variation_prob:
                variant = self._vary_thread(variant)
                if not isinstance(variant, dict):
                    print(f"⚠️  _vary_thread returned {type(variant)} instead of dict")
                    continue
                
            if random.random() < self.config.obfuscation_prob:
                variant = self._add_obfuscation(variant)
                if not isinstance(variant, dict):
                    print(f"⚠️  _add_obfuscation returned {type(variant)} instead of dict")
                    continue
            
            # Update metadata to track augmentation (handle both _metadata and meta formats)
            metadata_key = "_metadata" if "_metadata" in variant else "meta"
            if metadata_key not in variant:
                variant[metadata_key] = {}
            variant[metadata_key]["augmented"] = True
            variant[metadata_key]["base_id"] = self._generate_id(base_example)
            variant[metadata_key]["variant_id"] = i + 1
            
            variants.append(variant)
        
        return variants
    
    def _vary_recipients(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create recipient variations to simulate different risk scenarios."""
        original_recipients = example.get("recipients", [])
        if not original_recipients:
            return example
        
        variation_type = random.choice([
            "personal_risk",      # Change to personal email domains
            "external_business",  # Change to external business domains  
            "mixed_recipients",   # Mix internal and external
            "additional_cc"       # Add more recipients
        ])
        
        if variation_type == "personal_risk":
            # High risk: change to personal email domains
            new_recipients = []
            for recipient in original_recipients[:2]:  # Limit to avoid spam
                username = recipient.split("@")[0] if "@" in recipient else "user"
                domain = random.choice(self.config.external_personal_domains)
                new_recipients.append(f"{username}@{domain}")
            example["recipients"] = new_recipients
            
        elif variation_type == "external_business":
            # Medium risk: external business partners
            new_recipients = []
            for recipient in original_recipients[:2]:
                username = recipient.split("@")[0] if "@" in recipient else "contact"
                domain = random.choice(self.config.external_business_domains)
                new_recipients.append(f"{username}@{domain}")
            example["recipients"] = new_recipients
            
        elif variation_type == "mixed_recipients":
            # Mixed risk scenario
            new_recipients = original_recipients[:1]  # Keep one original
            username = random.choice(self.email_prefixes)
            domain = random.choice(self.config.external_personal_domains)
            new_recipients.append(f"{username}@{domain}")
            example["recipients"] = new_recipients
            
        elif variation_type == "additional_cc":
            # Add more recipients for complexity
            additional_count = random.randint(1, 2)
            new_recipients = original_recipients[:]
            for _ in range(additional_count):
                username = random.choice(self.email_prefixes)
                domain = random.choice(self.config.external_business_domains)
                new_recipients.append(f"{username}@{domain}")
            example["recipients"] = new_recipients
        
        return example
    
    def _vary_content(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create content variations while preserving sensitive information."""
        body = example.get("body", "")
        
        # Vary names in the content
        for original_name in self.name_pool[:5]:  # Check first few names
            if original_name in body:
                replacement_name = random.choice(self.name_pool)
                body = body.replace(original_name, replacement_name)
        
        # Vary some common business terms
        replacements = {
            "Q3": random.choice(["Q1", "Q2", "Q4"]),
            "2024": random.choice(["2023", "2025"]),
            "Monday": random.choice(["Tuesday", "Wednesday", "Thursday", "Friday"]),
            "next week": random.choice(["this week", "next month", "soon"]),
            "urgent": random.choice(["important", "critical", "time-sensitive"])
        }
        
        for old_term, new_term in replacements.items():
            if old_term in body:
                body = body.replace(old_term, new_term)
        
        # Vary subject line slightly
        subject = example.get("subject", "")
        if "Re:" not in subject and random.random() < 0.3:
            example["subject"] = f"Re: {subject}"
        
        example["body"] = body
        return example
    
    def _vary_thread(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create thread variations for multi-turn conversation simulation."""
        thread = example.get("thread", {})
        
        # Convert single-turn to multi-turn sometimes
        if thread.get("prior_msgs", 0) == 0 and random.random() < 0.5:
            thread["prior_msgs"] = random.randint(1, 4)
            thread["age_days"] = random.randint(1, 7)
        
        # Or vary existing multi-turn
        elif thread.get("prior_msgs", 0) > 0:
            thread["prior_msgs"] = random.randint(1, 6)
            thread["age_days"] = random.randint(0, 14)
        
        # Update thread ID to reflect variation
        thread["id_hash"] = self._generate_thread_id()
        
        example["thread"] = thread
        return example
    
    def _add_obfuscation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add obfuscation patterns to simulate evasion attempts."""
        body = example.get("body", "")
        
        obfuscation_types = [
            "asterisk_masking",   # Replace chars with asterisks
            "base64_encoding",    # Add base64 encoded content
            "character_insertion", # Insert zero-width chars
            "homoglyph_replacement" # Replace with similar looking chars
        ]
        
        obfuscation = random.choice(obfuscation_types)
        
        if obfuscation == "asterisk_masking":
            # Add masked sensitive patterns
            patterns = [
                (r"(\d{3})-(\d{2})-(\d{4})", r"***-**-\3"),  # SSN masking
                (r"(\d{4})\s*(\d{4})\s*(\d{4})\s*(\d{4})", r"****-****-****-\4"),  # Card masking
            ]
            for pattern, replacement in patterns:
                body = re.sub(pattern, replacement, body)
                
        elif obfuscation == "base64_encoding":
            # Add some base64 encoded content
            sensitive_phrases = ["confidential", "password", "secret", "private"]
            for phrase in sensitive_phrases:
                if phrase in body.lower():
                    import base64
                    encoded = base64.b64encode(phrase.encode()).decode()
                    body += f"\n\nEncoded reference: {encoded}"
                    break
        
        elif obfuscation == "character_insertion":
            # Insert zero-width characters in sensitive patterns
            # (Simplified - just add unusual spacing)
            body = re.sub(r"(\d{3})-(\d{2})-(\d{4})", r"\1-\2-\3 ", body)
        
        elif obfuscation == "homoglyph_replacement":
            # Replace some characters with look-alikes
            replacements = {"o": "0", "e": "3", "a": "@", "i": "1"}
            for old_char, new_char in replacements.items():
                if random.random() < 0.1:  # Low probability to avoid breaking readability
                    body = body.replace(old_char, new_char, 1)  # Replace only first occurrence
        
        example["body"] = body
        return example
    
    def _generate_id(self, example: Dict[str, Any]) -> str:
        """Generate unique ID for example."""
        content = json.dumps(example, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_thread_id(self) -> str:
        """Generate unique thread ID."""
        import time
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def augment_batch(self, base_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Augment a batch of base examples with validation."""
        all_variants = []
        
        for i, base_example in enumerate(base_examples):
            # Validate input
            if not isinstance(base_example, dict):
                print(f"⚠️  Skipping non-dict base example at index {i}: {type(base_example)}")
                continue
            
            try:
                # Add the original example
                original = deepcopy(base_example)
                # Handle both _metadata and meta field formats
                metadata_field = "_metadata" if "_metadata" in original else "meta"
                if metadata_field not in original:
                    original[metadata_field] = {}
                original[metadata_field]["augmented"] = False
                all_variants.append(original)
                
                # Add augmented variants
                variants = self.augment_example(base_example)
                
                # Validate that all variants are dicts
                for j, variant in enumerate(variants):
                    if isinstance(variant, dict):
                        all_variants.append(variant)
                    else:
                        print(f"⚠️  Skipping non-dict variant {j} from example {i}: {type(variant)}")
                        if isinstance(variant, str):
                            print(f"      String content sample: {variant[:100]}...")
                        
            except Exception as e:
                print(f"⚠️  Error augmenting example {i}: {e}")
                continue
        
        print(f"   📊 Augmentation: {len(base_examples)} → {len(all_variants)} examples")
        return all_variants


class RiskScenarioGenerator:
    """Generate specific risk scenarios through targeted augmentation."""
    
    def __init__(self, augmentor: DatasetAugmentor):
        self.augmentor = augmentor
    
    def create_violation_scenarios(self, authorized_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert authorized examples to violation scenarios."""
        violation_examples = []
        
        for example in authorized_examples:
            if example.get("meta", {}).get("risk_type") == "authorized":
                violation = deepcopy(example)
                
                # Change recipients to personal domains for violation
                recipients = violation.get("recipients", [])
                violation_recipients = []
                
                for recipient in recipients:
                    username = recipient.split("@")[0] if "@" in recipient else "user"
                    personal_domain = random.choice(self.augmentor.config.external_personal_domains)
                    violation_recipients.append(f"{username}@{personal_domain}")
                
                violation["recipients"] = violation_recipients
                violation["meta"]["risk_type"] = "violation"
                violation["meta"]["violation_reason"] = "personal_email_exposure"
                
                violation_examples.append(violation)
        
        return violation_examples
    
    def create_obfuscated_scenarios(self, base_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create obfuscated versions of examples."""
        obfuscated_examples = []
        
        for example in base_examples:
            obfuscated = deepcopy(example)
            obfuscated = self.augmentor._add_obfuscation(obfuscated)
            
            obfuscated["meta"]["risk_type"] = "obfuscated"
            obfuscated["meta"]["obfuscation_applied"] = True
            
            obfuscated_examples.append(obfuscated)
        
        return obfuscated_examples