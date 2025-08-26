"""Synthetic DLP Dataset Generator

Generates training, validation, and test datasets for HRM-DLP with:
- Realistic email/chat/PR scenarios
- PII, secrets, and other sensitive spans
- Document-level labels based on heuristics
- Augmentations and hard negatives
"""

import json
import os
import random
import re
import base64
import string
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import uuid
import hashlib


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation"""
    output_dir: str = "data/dlp_synth"
    train_size: int = 60000
    val_size: int = 5000
    test_size: int = 5000
    seed: int = 42
    
    # Augmentation probabilities
    obfuscation_prob: float = 0.15
    lookalike_prob: float = 0.1
    hard_negative_prob: float = 0.2
    
    # Domain separation
    use_domain_separation: bool = True


class SpanGenerator:
    """Generates realistic spans for different types of sensitive information"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize patterns and dictionaries for span generation"""
        # Email patterns
        self.email_domains = [
            "company.com", "corp.com", "enterprise.com", "business.com",
            "gmail.com", "proton.me", "outlook.com", "yahoo.com",
            "smith-legal.pt", "jones-law.com", "counsel-partners.com"
        ]
        
        # Phone patterns
        self.phone_prefixes = ["+1", "+44", "+351", "+49", "+33"]
        
        # Common names
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Maria",
            "James", "Emma", "William", "Olivia", "Daniel", "Sophia", "Matthew", "Isabella"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
            "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Garcia", "Martinez"
        ]
        
        # Company/legal terms
        self.legal_terms = [
            "NDA", "Non-Disclosure Agreement", "Confidentiality Agreement", 
            "Mutual NDA", "Engagement Letter", "Retainer Agreement"
        ]
        
        self.matter_prefixes = ["M", "MAT", "CASE", "PROJ"]
        
        # Secret patterns
        self.secret_prefixes = [
            "AKIA", "SK_", "xoxp-", "xoxb-", "ghp_", "github_pat_", "ya29.",
            "AIza", "gho_", "ghu_", "ghs_", "ghr_"
        ]
        
        # Database URI patterns
        self.db_schemes = [
            "postgresql://", "mysql://", "mongodb://", "redis://", 
            "postgres://", "sqlite://", "mssql://", "oracle://"
        ]
    
    def generate_email(self) -> str:
        """Generate realistic email address"""
        name = random.choice(self.first_names).lower()
        surname = random.choice(self.last_names).lower()
        domain = random.choice(self.email_domains)
        
        patterns = [
            f"{name}.{surname}@{domain}",
            f"{name}@{domain}",
            f"{name[0]}{surname}@{domain}",
            f"{name}.{surname[0]}@{domain}"
        ]
        
        return random.choice(patterns)
    
    def generate_phone(self) -> str:
        """Generate realistic phone number"""
        prefix = random.choice(self.phone_prefixes)
        if prefix == "+1":
            # US format
            area = random.randint(201, 999)
            exchange = random.randint(201, 999)
            number = random.randint(1000, 9999)
            return f"{prefix} ({area}) {exchange}-{number}"
        else:
            # International format
            number = ''.join([str(random.randint(0, 9)) for _ in range(9)])
            return f"{prefix} {number[:3]} {number[3:6]} {number[6:]}"
    
    def generate_pan(self) -> str:
        """Generate realistic PAN (credit card) number with Luhn check"""
        # Common prefixes: Visa (4), MasterCard (5), Amex (34, 37)
        prefixes = ["4", "5", "34", "37"]
        prefix = random.choice(prefixes)
        
        if prefix in ["34", "37"]:
            # Amex: 15 digits
            digits = [int(d) for d in prefix] + [random.randint(0, 9) for _ in range(13)]
        else:
            # Visa/MC: 16 digits
            digits = [int(prefix)] + [random.randint(0, 9) for _ in range(15)]
        
        # Apply Luhn algorithm
        def luhn_check(digits):
            total = 0
            for i, digit in enumerate(reversed(digits[:-1])):
                if i % 2 == 0:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                total += digit
            return (10 - (total % 10)) % 10
        
        digits[-1] = luhn_check(digits)
        
        # Format with spaces
        if len(digits) == 15:
            return f"{digits[0]}{digits[1]} {''.join(map(str, digits[2:8]))} {''.join(map(str, digits[8:]))}"
        else:
            return f"{''.join(map(str, digits[:4]))} {''.join(map(str, digits[4:8]))} {''.join(map(str, digits[8:12]))} {''.join(map(str, digits[12:]))}"
    
    def generate_ssn(self) -> str:
        """Generate SSN-like identifier"""
        # US SSN format: XXX-XX-XXXX
        area = random.randint(100, 899)  # Avoid invalid ranges
        group = random.randint(10, 99)
        serial = random.randint(1000, 9999)
        return f"{area:03d}-{group:02d}-{serial:04d}"
    
    def generate_secret_key(self) -> str:
        """Generate realistic API key or token"""
        prefix = random.choice(self.secret_prefixes)
        
        if prefix == "-----BEGIN":
            # PEM format
            key_type = random.choice(["PRIVATE KEY", "RSA PRIVATE KEY", "CERTIFICATE"])
            return f"-----BEGIN {key_type}-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC..."
        elif prefix in ["AKIA", "SK_"]:
            # AWS-style keys
            suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            return f"{prefix}{suffix}"
        elif prefix.startswith("xox"):
            # Slack-style tokens
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=24))
            return f"{prefix}-{suffix}"
        else:
            # GitHub/generic tokens
            suffix = ''.join(random.choices(string.ascii_letters + string.digits + "_", k=32))
            return f"{prefix}{suffix}"
    
    def generate_db_uri(self) -> str:
        """Generate database connection URI"""
        scheme = random.choice(self.db_schemes)
        user = random.choice(["admin", "user", "app", "service"])
        password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        host = random.choice(["localhost", "db.company.com", "prod-db-01", "staging.db.internal"])
        port = random.choice([5432, 3306, 27017, 6379, 1433])
        db_name = random.choice(["production", "staging", "users", "analytics", "logs"])
        
        return f"{scheme}{user}:{password}@{host}:{port}/{db_name}"
    
    def generate_nda_term(self) -> str:
        """Generate legal NDA reference"""
        term = random.choice(self.legal_terms)
        if random.choice([True, False]):
            # Add agreement number
            num = random.randint(1000, 9999)
            return f"{term} #{num}"
        return term
    
    def generate_matter_id(self) -> str:
        """Generate legal matter ID"""
        prefix = random.choice(self.matter_prefixes)
        number = random.randint(1000, 99999)
        return f"{prefix}-{number}"
    
    def generate_name(self) -> str:
        """Generate full name"""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        return f"{first} {last}"
    
    def generate_address(self) -> str:
        """Generate mailing address"""
        number = random.randint(100, 9999)
        streets = ["Main St", "Oak Ave", "First St", "Park Rd", "Elm St", "Washington Ave"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Boston"]
        
        street = random.choice(streets)
        city = random.choice(cities)
        zip_code = random.randint(10000, 99999)
        
        return f"{number} {street}, {city} {zip_code}"


class ContentGenerator:
    """Generates realistic email/chat content with embedded spans"""
    
    def __init__(self, span_generator: SpanGenerator):
        self.span_gen = span_generator
        self._init_templates()
    
    def _init_templates(self):
        """Initialize content templates"""
        self.roles = ["LEGAL", "FINANCE", "HR", "ENG", "MARKETING", "INTERN"]
        self.departments = ["CORP", "LEGAL", "TECH", "SALES", "HR"]
        self.seniorities = ["JUNIOR", "SENIOR", "PRINCIPAL", "DIRECTOR", "VP"]
        
        self.channels = ["email", "chat", "pr", "upload"]
        
        # Email templates
        self.email_templates = {
            "legal_formal": {
                "subjects": [
                    "RE: {matter_id} - Document Review",
                    "Portugal HQ - Legal Documentation",
                    "{nda_term} - Execution Required",
                    "Confidential: {matter_id} Update"
                ],
                "bodies": [
                    "Per the {nda_term} and {matter_id}, please find attached the requested documentation. The client information includes {sensitive_data}. Please handle with appropriate confidentiality measures.",
                    "Following our discussion regarding {matter_id}, I am providing the sensitive client data: {sensitive_data}. This information is subject to the {nda_term} executed on the matter.",
                    "The legal review for {matter_id} requires access to {sensitive_data}. Please ensure this information remains confidential per our {nda_term}."
                ]
            },
            "finance_transactional": {
                "subjects": [
                    "Q3 Financial Report - CONFIDENTIAL",
                    "Payment Processing Update",
                    "Customer Payment Data",
                    "Invoice #{invoice_num} - Payment Required"
                ],
                "bodies": [
                    "Please process the payment using the following details: {sensitive_data}. The transaction should be completed by end of business today.",
                    "The customer payment information is as follows: {sensitive_data}. Please ensure PCI compliance in handling this data.",
                    "Attached is the financial report containing customer data including {sensitive_data}. This information is strictly confidential."
                ]
            },
            "marketing_casual": {
                "subjects": [
                    "Customer List for Campaign",
                    "User Analytics Data",
                    "Marketing Campaign Results",
                    "Customer Outreach - Action Required"
                ],
                "bodies": [
                    "Hi team! Here's the customer data for the new campaign: {sensitive_data}. Let me know if you need anything else!",
                    "Quick update on the campaign - we have the following customer information: {sensitive_data}. Looking good so far!",
                    "The latest user analytics show {sensitive_data}. This should help with our targeting strategy."
                ]
            }
        }
        
        # Attachment templates
        self.attachment_templates = [
            {"name": "customers_q3.csv", "mime": "text/csv", "size": 82000},
            {"name": "financial_report.pdf", "mime": "application/pdf", "size": 156000},
            {"name": "user_data.xlsx", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "size": 234000},
            {"name": "config.json", "mime": "application/json", "size": 12000},
            {"name": "database_backup.sql", "mime": "application/sql", "size": 1500000}
        ]
        
        # Link templates
        self.link_templates = [
            "https://share.company.com/financial-reports/q3-2024",
            "https://drive.google.com/shared/confidential-docs",
            "https://dropbox.com/s/abc123/sensitive-data.pdf",
            "https://company.sharepoint.com/legal/matter-files",
            "https://bit.ly/shortened-link-123"
        ]
    
    def generate_example(self, with_spans: bool = True, obfuscate: bool = False) -> Dict[str, Any]:
        """Generate a complete DLP example"""
        # Select role and tone
        role = random.choice(self.roles)
        channel = random.choice(self.channels)
        
        # Select template based on role
        if role == "LEGAL":
            template_key = "legal_formal"
        elif role == "FINANCE":
            template_key = "finance_transactional"
        else:
            template_key = "marketing_casual"
        
        template = self.email_templates[template_key]
        
        # Generate content
        spans = []
        sensitive_data_parts = []
        
        if with_spans:
            # Generate different types of sensitive data
            span_types = ["EMAIL", "PHONE", "PAN", "SSN", "SECRET", "DBURI", "NDA", "MATTER", "NAME", "ADDR"]
            num_spans = random.randint(1, 4)
            
            selected_types = random.sample(span_types, min(num_spans, len(span_types)))
            
            for span_type in selected_types:
                if span_type == "EMAIL":
                    data = self.span_gen.generate_email()
                elif span_type == "PHONE":
                    data = self.span_gen.generate_phone()
                elif span_type == "PAN":
                    data = self.span_gen.generate_pan()
                elif span_type == "SSN":
                    data = self.span_gen.generate_ssn()
                elif span_type == "SECRET":
                    data = self.span_gen.generate_secret_key()
                elif span_type == "DBURI":
                    data = self.span_gen.generate_db_uri()
                elif span_type == "NDA":
                    data = self.span_gen.generate_nda_term()
                elif span_type == "MATTER":
                    data = self.span_gen.generate_matter_id()
                elif span_type == "NAME":
                    data = self.span_gen.generate_name()
                elif span_type == "ADDR":
                    data = self.span_gen.generate_address()
                
                sensitive_data_parts.append(data)
        
        # Create body content
        subject_template = random.choice(template["subjects"])
        body_template = random.choice(template["bodies"])
        
        # Fill template variables
        nda_term = self.span_gen.generate_nda_term()
        matter_id = self.span_gen.generate_matter_id()
        invoice_num = random.randint(10000, 99999)
        sensitive_data = ", ".join(sensitive_data_parts) if sensitive_data_parts else "customer information"
        
        subject = subject_template.format(
            nda_term=nda_term,
            matter_id=matter_id,
            invoice_num=invoice_num
        )
        
        body = body_template.format(
            nda_term=nda_term,
            matter_id=matter_id,
            sensitive_data=sensitive_data,
            invoice_num=invoice_num
        )
        
        # Apply obfuscation if requested
        if obfuscate:
            body = self._apply_obfuscation(body)
        
        # Find spans in the final body text
        body_spans = self._extract_spans(body)
        
        # Generate recipients
        recipients = self._generate_recipients(role)
        
        # Generate other metadata
        user_info = {
            "role": role,
            "dept": random.choice(self.departments),
            "seniority": random.choice(self.seniorities),
            "id_hash": self._hash_string(f"user_{random.randint(1000, 9999)}")
        }
        
        thread_info = {
            "id_hash": self._hash_string(f"thread_{random.randint(1000, 9999)}"),
            "age_days": random.randint(0, 30),
            "prior_msgs": random.randint(0, 10)
        }
        
        # Generate attachments and links
        attachments = []
        links = []
        
        if random.random() < 0.3:  # 30% chance of attachments
            num_attachments = random.randint(1, 2)
            attachments = random.sample(self.attachment_templates, min(num_attachments, len(self.attachment_templates)))
        
        if random.random() < 0.2:  # 20% chance of links
            num_links = random.randint(1, 2)
            links = random.sample(self.link_templates, min(num_links, len(self.link_templates)))
        
        # Generate labels
        labels = self._generate_labels(body_spans, recipients, role, obfuscate)
        
        # Create meta information
        meta = {
            "base64": obfuscate and "base64" in body.lower(),
            "homoglyph": obfuscate and any(ord(c) > 127 for c in body),
            "ts": self._generate_timestamp()
        }
        
        return {
            "channel": channel,
            "user": user_info,
            "recipients": recipients,
            "thread": thread_info,
            "subject": subject,
            "body": body,
            "attachments": attachments,
            "links": links,
            "labels": labels,
            "spans": body_spans,
            "meta": meta
        }
    
    def _apply_obfuscation(self, text: str) -> str:
        """Apply various obfuscation techniques"""
        techniques = ["base64", "homoglyph", "zero_width"]
        technique = random.choice(techniques)
        
        if technique == "base64":
            # Base64 encode random parts
            words = text.split()
            if words:
                word_to_encode = random.choice(words)
                encoded = base64.b64encode(word_to_encode.encode()).decode()
                text = text.replace(word_to_encode, f"[BASE64:{encoded}]")
        
        elif technique == "homoglyph":
            # Replace some characters with lookalikes
            replacements = {
                'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
                'x': 'х', 'y': 'у', 'B': 'В', 'H': 'Н', 'K': 'К'
            }
            for original, replacement in replacements.items():
                if random.random() < 0.3:
                    text = text.replace(original, replacement)
        
        elif technique == "zero_width":
            # Insert zero-width characters
            zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
            char_positions = random.sample(range(len(text)), min(5, len(text)))
            for pos in sorted(char_positions, reverse=True):
                text = text[:pos] + random.choice(zero_width_chars) + text[pos:]
        
        return text
    
    def _extract_spans(self, text: str) -> List[Dict[str, Any]]:
        """Extract spans from text using regex patterns"""
        spans = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            spans.append({
                "type": "EMAIL",
                "start": match.start(),
                "end": match.end()
            })
        
        # Phone pattern
        phone_pattern = r'(\+\d{1,3}\s?)?(\(\d{3}\)\s?)?[\d\s\-\.]{10,}'
        for match in re.finditer(phone_pattern, text):
            spans.append({
                "type": "PHONE", 
                "start": match.start(),
                "end": match.end()
            })
        
        # PAN pattern (credit card)
        pan_pattern = r'\b(?:\d{4}\s){3}\d{4}\b|\b\d{2}\s\d{6}\s\d{7}\b'
        for match in re.finditer(pan_pattern, text):
            spans.append({
                "type": "PAN",
                "start": match.start(),
                "end": match.end()
            })
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            spans.append({
                "type": "SSN",
                "start": match.start(),
                "end": match.end()
            })
        
        # Secret key patterns
        secret_patterns = [
            r'\bAKIA[A-Z0-9]{16}\b',
            r'\bSK_[A-Za-z0-9_]{16,}\b',
            r'\bxoxp-[a-z0-9\-]{24,}\b',
            r'\bghp_[A-Za-z0-9_]{36}\b',
            r'-----BEGIN [A-Z\s]+-----'
        ]
        
        for pattern in secret_patterns:
            for match in re.finditer(pattern, text):
                spans.append({
                    "type": "SECRET",
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Database URI pattern
        db_pattern = r'\b(?:postgresql|mysql|mongodb|redis|postgres|sqlite|mssql|oracle)://[^\s]+\b'
        for match in re.finditer(db_pattern, text):
            spans.append({
                "type": "DBURI",
                "start": match.start(),
                "end": match.end()
            })
        
        # NDA terms
        nda_pattern = r'\b(?:NDA|Non-Disclosure|Confidentiality Agreement|Mutual NDA|Engagement|Agreement\s*#\d+)\b'
        for match in re.finditer(nda_pattern, text, re.IGNORECASE):
            spans.append({
                "type": "NDA",
                "start": match.start(),
                "end": match.end()
            })
        
        # Matter ID pattern
        matter_pattern = r'\b(?:M|MAT|CASE|PROJ)-\d{3,5}\b'
        for match in re.finditer(matter_pattern, text):
            spans.append({
                "type": "MATTER",
                "start": match.start(),
                "end": match.end()
            })
        
        return spans
    
    def _generate_recipients(self, role: str) -> List[str]:
        """Generate appropriate recipients based on role"""
        if role == "LEGAL":
            # Legal communications go to legal domains or internal
            domains = ["smith-legal.pt", "jones-law.com", "counsel-partners.com", "company.com"]
        elif role in ["FINANCE", "HR"]:
            # Internal or trusted business partners
            domains = ["company.com", "corp.com", "enterprise.com", "partner-firm.com"]
        else:
            # Mix of internal and external (including personal)
            domains = ["company.com", "gmail.com", "outlook.com", "proton.me", "partner.com"]
        
        num_recipients = random.randint(1, 3)
        recipients = []
        
        for _ in range(num_recipients):
            name = random.choice(self.span_gen.first_names).lower()
            surname = random.choice(self.span_gen.last_names).lower()
            domain = random.choice(domains)
            email = f"{name}.{surname}@{domain}"
            recipients.append(email)
        
        return recipients
    
    def _generate_labels(self, spans: List[Dict], recipients: List[str], role: str, obfuscated: bool) -> Dict[str, int]:
        """Generate document-level labels based on heuristics"""
        labels = {
            "sensitivity": 0,
            "exposure": 0,
            "context": 0,
            "obfuscation": 0
        }
        
        # Sensitivity: presence of PII/secrets
        sensitive_types = {"PAN", "SECRET", "DBURI", "SSN"}
        has_sensitive = any(span["type"] in sensitive_types for span in spans)
        has_pii = any(span["type"] in {"NAME", "EMAIL", "PHONE", "ADDR"} for span in spans)
        
        if has_sensitive or (has_pii and len([s for s in spans if s["type"] in {"NAME", "EMAIL"}]) > 1):
            labels["sensitivity"] = 1
        
        # Exposure: risk based on recipients
        external_personal_domains = {"gmail.com", "proton.me", "outlook.com", "yahoo.com"}
        has_external_personal = any(any(domain in recipient for domain in external_personal_domains) for recipient in recipients)
        
        if has_external_personal:
            labels["exposure"] = 1
        
        # Context: legitimate workflow indicators
        has_nda = any(span["type"] == "NDA" for span in spans)
        has_matter = any(span["type"] == "MATTER" for span in spans)
        is_legal_role = role == "LEGAL"
        
        if (is_legal_role and (has_nda or has_matter)) or (has_nda and not has_external_personal):
            labels["context"] = 1
        
        # Obfuscation: detected obfuscation techniques
        if obfuscated:
            labels["obfuscation"] = 1
        
        return labels
    
    def _hash_string(self, s: str) -> str:
        """Create hash for IDs"""
        return hashlib.md5(s.encode()).hexdigest()[:8]
    
    def _generate_timestamp(self) -> str:
        """Generate realistic timestamp"""
        base_time = datetime.now() - timedelta(days=random.randint(0, 365))
        return base_time.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_dataset(config: DataGenConfig) -> None:
    """Generate complete synthetic dataset"""
    print(f"Generating synthetic DLP dataset with seed {config.seed}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize generators
    span_gen = SpanGenerator(config.seed)
    content_gen = ContentGenerator(span_gen)
    
    # Generate splits
    splits = [
        ("train", config.train_size),
        ("val", config.val_size),
        ("test", config.test_size)
    ]
    
    for split_name, split_size in splits:
        print(f"Generating {split_name} split ({split_size} examples)...")
        
        output_file = os.path.join(config.output_dir, f"{split_name}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(split_size):
                # Apply augmentations probabilistically
                obfuscate = random.random() < config.obfuscation_prob
                
                # Generate example
                example = content_gen.generate_example(
                    with_spans=True,
                    obfuscate=obfuscate
                )
                
                # Write to file
                f.write(json.dumps(example) + "\n")
                
                if (i + 1) % 1000 == 0:
                    print(f"  Generated {i + 1}/{split_size} examples")
        
        print(f"Saved {split_name} split to {output_file}")
    
    # Generate data statistics
    print("\nDataset generation complete!")
    print(f"Files saved to: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DLP training data")
    parser.add_argument("--output-dir", type=str, default="data/dlp_synth", 
                       help="Output directory for generated data")
    parser.add_argument("--train-size", type=int, default=60000,
                       help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=5000,
                       help="Number of validation examples")
    parser.add_argument("--test-size", type=int, default=5000,
                       help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--obfuscation-prob", type=float, default=0.15,
                       help="Probability of applying obfuscation")
    
    args = parser.parse_args()
    
    config = DataGenConfig(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        obfuscation_prob=args.obfuscation_prob
    )
    
    generate_dataset(config)


if __name__ == "__main__":
    main()