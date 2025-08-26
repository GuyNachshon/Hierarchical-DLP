"""LLM-based Synthetic Data Generator for HRM-DLP

Uses large language models to generate realistic email/chat scenarios with:
- Natural language content
- Realistic sensitive information placement
- Diverse communication patterns
- Proper labeling and span extraction
"""

import json
import os
import re
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import argparse
import asyncio
from datetime import datetime, timedelta
import hashlib

import openai
from anthropic import Anthropic
import requests


@dataclass
class LLMGeneratorConfig:
    """Configuration for LLM-based data generation"""
    # Output settings
    output_dir: str = "data/dlp_llm_synth"
    train_size: int = 60000
    val_size: int = 5000
    test_size: int = 5000
    
    # LLM settings
    llm_provider: str = "openai"  # "openai", "anthropic", "local"
    model_name: str = "gpt-4o-mini"  # Cost-effective for data generation
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local models
    
    # Generation parameters
    batch_size: int = 10  # Generate multiple examples per API call
    max_retries: int = 3
    temperature: float = 0.8
    
    # Quality control
    min_body_length: int = 50
    max_body_length: int = 2000
    min_spans_per_example: int = 1
    max_spans_per_example: int = 5
    
    # Data diversity
    seed: int = 42


class LLMDataGenerator:
    """Generate synthetic DLP data using LLMs"""
    
    def __init__(self, config: LLMGeneratorConfig):
        self.config = config
        self.client = self._init_llm_client()
        self._init_prompt_templates()
        
        # Track generated domains/emails for deduplication
        self.used_emails: Set[str] = set()
        self.generation_stats = {
            "total_generated": 0,
            "successful": 0,
            "failed_parsing": 0,
            "failed_validation": 0
        }
    
    def _init_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.config.llm_provider == "openai":
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key")
            
            return openai.OpenAI(
                api_key=api_key,
                base_url=self.config.base_url
            )
        
        elif self.config.llm_provider == "anthropic":
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var")
            
            return Anthropic(api_key=api_key)
        
        elif self.config.llm_provider == "local":
            # For local models (e.g., via vLLM, ollama)
            class LocalClient:
                def __init__(self, base_url):
                    self.base_url = base_url or "http://localhost:8000"
                
                def chat_completions_create(self, **kwargs):
                    # Implement local API call
                    pass
            
            return LocalClient(self.config.base_url)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def _init_prompt_templates(self):
        """Initialize generation statistics tracking"""
        # No longer using templates - LLM generates everything dynamically
        pass
    
    def _generate_with_llm(self, prompt: str, system: str) -> Optional[str]:
        """Generate content using configured LLM"""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.llm_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=1500
                    )
                    return response.choices[0].message.content
                
                elif self.config.llm_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=1500,
                        temperature=self.config.temperature,
                        system=system,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                
                else:
                    # Local provider implementation
                    pass
                    
            except Exception as e:
                print(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract single JSON from LLM response"""
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON code blocks
        json_blocks = re.findall(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        for block in json_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extract_multiple_json(self, response: str) -> List[Dict]:
        """Extract multiple JSON objects from LLM response"""
        examples = []
        
        # Try line-by-line parsing first (preferred format)
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # If line-by-line didn't work, try regex extraction
        if not examples:
            # Find all JSON objects in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    example = json.loads(match)
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # Final fallback: try to parse the whole response as JSON array
        if not examples:
            try:
                # Check if it's a JSON array
                if response.strip().startswith('[') and response.strip().endswith(']'):
                    examples = json.loads(response.strip())
            except json.JSONDecodeError:
                pass
        
        return examples
    
    def _generate_metadata_with_llm(self, scenario_type: str) -> Dict[str, Any]:
        """Generate realistic user info, recipients, and metadata using LLM"""
        
        metadata_prompt = f"""Generate realistic corporate email metadata for a {scenario_type} scenario.

Create diverse, realistic:
- User roles (mix of legal, finance, engineering, marketing, HR, etc.)
- Departments and seniority levels
- Company domains (be creative - use realistic company names)
- Recipient emails that match the scenario risk level
- Realistic names and email formats

For {scenario_type}:
{self._get_scenario_guidance(scenario_type)}

Generate JSON with this EXACT format:
{{
  "user": {{
    "role": "ENGINEERING", 
    "dept": "TECH",
    "seniority": "SENIOR"
  }},
  "sender_email": "realistic.name@company.com",
  "recipients": ["recipient1@domain.com", "recipient2@otherdomain.com"],
  "company_domain": "company.com",
  "channel": "email",
  "risk_context": "brief explanation of why this scenario fits {scenario_type}"
}}

Be creative with company names and domains - don't use generic ones."""
        
        system_prompt = "You are an expert at generating realistic corporate communication metadata. Create diverse, believable business scenarios."
        
        response = self._generate_with_llm(metadata_prompt, system_prompt)
        if response:
            metadata = self._extract_json_from_response(response)
            if metadata:
                return metadata
        
        # Fallback to rule-based if LLM fails
        return self._fallback_metadata_generation(scenario_type)
    
    def _get_scenario_guidance(self, scenario_type: str) -> str:
        """Get guidance for LLM based on scenario type"""
        guidance = {
            "legal_legitimate": """
- Recipients should be external legal counsel, other law firms, or internal legal team
- Use professional law firm domains (like smith-legal.com, jones-law.partners)
- High-trust, legitimate business relationship""",
            
            "data_leak": """
- Recipients should be personal email domains (gmail, outlook, proton)
- Or suspicious external domains unrelated to business
- Represents inappropriate data sharing""",
            
            "obfuscated_content": """
- Can be any recipient type
- Focus on creating suspicious communication patterns
- Someone trying to hide something""",
            
            "casual_internal": """
- Recipients should be same company domain or trusted business partners
- Normal workplace communication
- Low-risk, routine business"""
        }
        return guidance.get(scenario_type, "")
    
    def _fallback_metadata_generation(self, scenario_type: str) -> Dict[str, Any]:
        """Fallback metadata generation if LLM fails"""
        roles = ["LEGAL", "FINANCE", "HR", "ENG", "MARKETING", "SALES", "ADMIN"]
        departments = ["LEGAL", "CORP", "TECH", "SALES", "HR", "FINANCE"]
        seniorities = ["JUNIOR", "SENIOR", "PRINCIPAL", "DIRECTOR", "VP"]
        
        # Generate random company domain
        company_names = ["acme", "globex", "initech", "wayneent", "oscorp", "umbrella"]
        company_domain = f"{random.choice(company_names)}.com"
        
        return {
            "user": {
                "role": random.choice(roles),
                "dept": random.choice(departments), 
                "seniority": random.choice(seniorities)
            },
            "sender_email": f"user.name@{company_domain}",
            "recipients": [f"recipient@{'gmail.com' if scenario_type == 'data_leak' else company_domain}"],
            "company_domain": company_domain,
            "channel": random.choice(["email", "chat", "slack"]),
            "risk_context": f"Fallback generation for {scenario_type}"
        }
    
    def _validate_and_extract_spans(self, content: str) -> Tuple[List[Dict], bool]:
        """Extract and validate spans from generated content"""
        spans = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, content):
            spans.append({
                "type": "EMAIL",
                "start": match.start(),
                "end": match.end(),
                "text": match.group()
            })
        
        # Phone pattern - more flexible
        phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        for match in re.finditer(phone_pattern, content):
            spans.append({
                "type": "PHONE",
                "start": match.start(), 
                "end": match.end(),
                "text": match.group()
            })
        
        # Credit card pattern - more robust
        cc_pattern = r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
        for match in re.finditer(cc_pattern, content):
            # Simple Luhn check
            digits = [int(d) for d in match.group() if d.isdigit()]
            if self._luhn_check(digits):
                spans.append({
                    "type": "PAN",
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group()
                })
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
        for match in re.finditer(ssn_pattern, content):
            spans.append({
                "type": "SSN",
                "start": match.start(),
                "end": match.end(),
                "text": match.group()
            })
        
        # API key patterns
        api_key_patterns = [
            (r'\bAKIA[A-Z0-9]{16}\b', "SECRET"),
            (r'\bsk-[A-Za-z0-9]{48}\b', "SECRET"),
            (r'\bxoxp-[a-z0-9\-]{72}\b', "SECRET"),
            (r'\bghp_[A-Za-z0-9]{36}\b', "SECRET"),
            (r'\b[A-Za-z0-9]{32,}\b(?=.*[A-Z])(?=.*[a-z])(?=.*[0-9])', "SECRET")  # Generic high-entropy
        ]
        
        for pattern, span_type in api_key_patterns:
            for match in re.finditer(pattern, content):
                spans.append({
                    "type": span_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group()
                })
        
        # Database URI pattern
        db_pattern = r'\b(?:postgresql|mysql|mongodb|redis|sqlite)://[^\s]+\b'
        for match in re.finditer(db_pattern, content):
            spans.append({
                "type": "DBURI",
                "start": match.start(),
                "end": match.end(),
                "text": match.group()
            })
        
        # Legal terms
        legal_pattern = r'\b(?:NDA|Non-Disclosure|Confidentiality Agreement|Matter [A-Z]-\d+|Agreement #\d+)\b'
        for match in re.finditer(legal_pattern, content, re.IGNORECASE):
            spans.append({
                "type": "NDA" if "NDA" in match.group().upper() else "MATTER",
                "start": match.start(),
                "end": match.end(), 
                "text": match.group()
            })
        
        # Names (simple heuristic)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(name_pattern, content):
            # Avoid common false positives
            name = match.group()
            if not any(word in name.lower() for word in ["new york", "los angeles", "united states", "data loss"]):
                spans.append({
                    "type": "NAME",
                    "start": match.start(),
                    "end": match.end(),
                    "text": name
                })
        
        # Remove duplicates and sort by position
        unique_spans = []
        seen_positions = set()
        
        for span in sorted(spans, key=lambda x: x["start"]):
            pos_key = (span["start"], span["end"])
            if pos_key not in seen_positions:
                unique_spans.append({
                    "type": span["type"],
                    "start": span["start"],
                    "end": span["end"]
                })
                seen_positions.add(pos_key)
        
        # Validate span count
        valid = (self.config.min_spans_per_example <= len(unique_spans) <= self.config.max_spans_per_example)
        
        return unique_spans, valid
    
    def _luhn_check(self, digits: List[int]) -> bool:
        """Validate credit card number using Luhn algorithm"""
        def luhn_sum(digits):
            total = 0
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                total += digit
            return total
        
        return luhn_sum(digits) % 10 == 0
    
    def _generate_document_labels(self, example: Dict, scenario_type: str) -> Dict[str, int]:
        """Generate document-level labels based on scenario and content"""
        labels = {"sensitivity": 0, "exposure": 0, "context": 0, "obfuscation": 0}
        
        body = example.get("body", "")
        recipients = example.get("recipients", [])
        user = example.get("user", {})
        
        # Extract spans to inform labeling
        spans, _ = self._validate_and_extract_spans(body)
        
        # Sensitivity: Based on span types present
        high_sensitivity_types = {"PAN", "SSN", "SECRET", "DBURI"}
        medium_sensitivity_types = {"EMAIL", "PHONE", "NAME"}
        
        if any(span["type"] in high_sensitivity_types for span in spans):
            labels["sensitivity"] = 1
        elif len([s for s in spans if s["type"] in medium_sensitivity_types]) >= 2:
            labels["sensitivity"] = 1
        
        # Exposure: Based on recipients and scenario
        external_domains = {"gmail.com", "proton.me", "outlook.com", "yahoo.com", "hotmail.com"}
        has_external_personal = any(any(domain in recipient for domain in external_domains) for recipient in recipients)
        
        if scenario_type == "data_leak" or has_external_personal:
            labels["exposure"] = 1
        
        # Context: Based on scenario and legal indicators
        has_legal_context = any(span["type"] in {"NDA", "MATTER"} for span in spans)
        is_legal_role = user.get("role") == "LEGAL"
        
        if scenario_type == "legal_legitimate" or (has_legal_context and is_legal_role):
            labels["context"] = 1
        
        # Obfuscation: Based on scenario
        if scenario_type == "obfuscated_content":
            labels["obfuscation"] = 1
        elif "base64" in body.lower() or "encoded" in body.lower():
            labels["obfuscation"] = 1
        
        return labels
    
    def generate_batch(self, scenario_type: str, split: str, batch_size: int) -> List[Dict]:
        """Generate a batch of examples for a given scenario using full LLM generation"""
        examples = []
        
        # Generate complete examples with LLM (including all metadata)
        batch_prompt = self._create_batch_prompt(scenario_type, batch_size)
        
        response = self._generate_with_llm(batch_prompt, "You are an expert at generating realistic corporate communication scenarios for DLP training. Create diverse, believable examples.")
        
        if not response:
            self.generation_stats["failed_parsing"] += batch_size
            return []
        
        # Try to extract multiple JSON objects from response
        extracted_examples = self._extract_multiple_json(response)
        
        for example_data in extracted_examples:
            if not example_data:
                self.generation_stats["failed_parsing"] += 1
                continue
            
            # Validate basic structure
            if not self._validate_example(example_data):
                self.generation_stats["failed_validation"] += 1
                continue
            
            # Extract spans from body
            body = example_data.get("body", "")
            spans, spans_valid = self._validate_and_extract_spans(body)
            
            if not spans_valid:
                self.generation_stats["failed_validation"] += 1
                continue
            
            # Generate labels based on LLM output and extracted spans
            labels = self._generate_document_labels(example_data, scenario_type)
            
            # Create final structured example
            final_example = {
                "channel": example_data.get("channel", "email"),
                "user": example_data.get("user", {}),
                "recipients": example_data.get("recipients", []),
                "thread": {
                    "id_hash": self._hash_string(f"thread_{random.randint(1000, 9999)}"),
                    "age_days": random.randint(0, 30),
                    "prior_msgs": random.randint(0, 10)
                },
                "subject": example_data.get("subject", ""),
                "body": body,
                "attachments": example_data.get("attachments", []),
                "links": example_data.get("links", []),
                "labels": labels,
                "spans": spans,
                "meta": {
                    "base64": "base64" in body.lower(),
                    "homoglyph": any(ord(c) > 127 for c in body),
                    "ts": self._generate_timestamp(),
                    "scenario": scenario_type,
                    "llm_generated": True
                }
            }
            
            examples.append(final_example)
            self.generation_stats["successful"] += 1
        
        self.generation_stats["total_generated"] += batch_size
        return examples
    
    def _create_batch_prompt(self, risk_level: str, batch_size: int) -> str:
        """Create prompt for generating a batch of examples with open-ended generation"""
        
        return f"""Generate {batch_size} diverse, realistic corporate communication examples for DLP training.

Create a mix of communications with varying risk levels ({risk_level} focus):

COMPLETE CREATIVE FREEDOM:
- Invent any company names, domains, roles, departments
- Create any realistic business scenarios 
- Include various types of sensitive information naturally
- Use different communication channels and styles
- Make each example unique with different contexts

Risk level guidance for "{risk_level}":
- low_risk: Normal business communications that should be allowed
- medium_risk: Borderline cases requiring careful analysis  
- high_risk: Clear violations that should be blocked
- obfuscated: Hidden/disguised sensitive content

For each example, create realistic:
- Employee roles and company structure
- Business domains and email addresses
- Appropriate sensitive data (credit cards, SSNs, API keys, personal info, legal docs, etc.)
- Natural communication patterns and business context
- Varied communication channels (email, chat, slack, teams, etc.)

Return EXACTLY {batch_size} JSON objects, one per line:

{{"channel": "email", "user": {{"role": "FINANCE", "dept": "ACCOUNTING", "seniority": "MANAGER"}}, "recipients": ["external@partnerbank.com"], "subject": "Q3 Client Payment Processing", "body": "Hi Sarah, Per our service agreement, here are the client payment details for Q3 processing: Card 4532-1234-5678-9012, routing 021000021. Please process by EOD Friday.", "attachments": [], "links": []}}

Generate {batch_size} completely unique examples with maximum creativity and diversity. Each should feel like real workplace communication."""
    
    def _validate_example(self, example: Dict) -> bool:
        """Validate generated example meets requirements"""
        required_fields = ["body", "subject", "user", "recipients"]
        
        # Check required fields
        for field in required_fields:
            if field not in example or not example[field]:
                return False
        
        # Validate body length
        body = example["body"]
        if not (self.config.min_body_length <= len(body) <= self.config.max_body_length):
            return False
        
        # Check for realistic content (not Lorem ipsum, etc.)
        if "lorem ipsum" in body.lower() or "placeholder" in body.lower():
            return False
        
        return True
    
    def _hash_string(self, s: str) -> str:
        """Create hash for IDs"""
        return hashlib.md5(s.encode()).hexdigest()[:8]
    
    def _generate_timestamp(self) -> str:
        """Generate realistic timestamp"""
        base_time = datetime.now() - timedelta(days=random.randint(0, 365))
        return base_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def generate_dataset(self) -> None:
        """Generate complete dataset using LLMs"""
        print(f"Starting LLM-based dataset generation")
        print(f"Provider: {self.config.llm_provider}, Model: {self.config.model_name}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Define risk level distribution (let LLM decide everything else)
        risk_distribution = {
            "low_risk": 0.4,      # 40% normal business communications
            "medium_risk": 0.3,   # 30% borderline cases  
            "high_risk": 0.2,     # 20% clear violations
            "obfuscated": 0.1     # 10% hidden content
        }
        
        splits = [
            ("train", self.config.train_size),
            ("val", self.config.val_size),
            ("test", self.config.test_size)
        ]
        
        for split_name, split_size in splits:
            print(f"\nGenerating {split_name} split ({split_size} examples)...")
            
            output_file = os.path.join(self.config.output_dir, f"{split_name}.jsonl")
            examples_generated = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                while examples_generated < split_size:
                    # Determine risk level for this batch (LLM decides specific scenario)
                    risk_level = random.choices(
                        list(risk_distribution.keys()),
                        weights=list(risk_distribution.values()),
                        k=1
                    )[0]
                    
                    # Generate batch
                    remaining = split_size - examples_generated
                    current_batch_size = min(self.config.batch_size, remaining)
                    
                    print(f"  Generating {current_batch_size} {risk_level} examples...")
                    
                    batch_examples = self.generate_batch(risk_level, split_name, current_batch_size)
                    
                    # Write successful examples
                    for example in batch_examples:
                        f.write(json.dumps(example) + "\n")
                        examples_generated += 1
                        
                        if examples_generated % 100 == 0:
                            print(f"    Progress: {examples_generated}/{split_size}")
                        
                        if examples_generated >= split_size:
                            break
                    
                    # Small delay to be respectful to API
                    time.sleep(0.5)
            
            print(f"✅ Completed {split_name} split: {examples_generated} examples")
        
        # Print generation statistics
        print(f"\n{'='*60}")
        print("GENERATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total attempted: {self.generation_stats['total_generated']}")
        print(f"Successful: {self.generation_stats['successful']}")
        print(f"Failed parsing: {self.generation_stats['failed_parsing']}")
        print(f"Failed validation: {self.generation_stats['failed_validation']}")
        print(f"Success rate: {self.generation_stats['successful']/self.generation_stats['total_generated']*100:.1f}%")
        
        # Save generation stats
        stats_file = os.path.join(self.config.output_dir, "generation_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        print(f"\nDataset saved to: {self.config.output_dir}")
        print("Files created:")
        for split_name, _ in splits:
            file_path = os.path.join(self.config.output_dir, f"{split_name}.jsonl")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    line_count = sum(1 for _ in f)
                print(f"  {split_name}.jsonl: {line_count} examples")


def main():
    parser = argparse.ArgumentParser(description="Generate DLP training data using LLMs")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="data/dlp_llm_synth",
                       help="Output directory for generated data")
    parser.add_argument("--train-size", type=int, default=60000,
                       help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=5000,
                       help="Number of validation examples") 
    parser.add_argument("--test-size", type=int, default=5000,
                       help="Number of test examples")
    
    # LLM settings
    parser.add_argument("--llm-provider", choices=["openai", "anthropic", "local"], 
                       default="openai", help="LLM provider to use")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini",
                       help="Model name to use")
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")
    parser.add_argument("--base-url", type=str, help="Base URL for local models")
    
    # Generation settings
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Examples to generate per API call")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for LLM generation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Quick demo mode
    parser.add_argument("--demo", action="store_true",
                       help="Generate small demo dataset (1000 train, 200 val, 200 test)")
    
    args = parser.parse_args()
    
    # Demo mode overrides
    if args.demo:
        args.train_size = 1000
        args.val_size = 200
        args.test_size = 200
        args.output_dir = "data/dlp_demo_llm"
        print("🎮 Demo mode: generating small dataset for quick testing")
    
    # Create config
    config = LLMGeneratorConfig(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed
    )
    
    # Generate dataset
    generator = LLMDataGenerator(config)
    generator.generate_dataset()
    
    print("\n🎉 LLM-based dataset generation completed!")
    print(f"\nNext steps:")
    print(f"1. Train tokenizer: python -c \"from hrm_dlp.tokenizer import create_tokenizer; create_tokenizer(['{args.output_dir}/train.jsonl'], 'tokenizers/dlp_llm')\"")
    print(f"2. Train model: python pretrain_dlp.py data_path={args.output_dir}")
    print(f"3. Evaluate: python evaluate_dlp.py --checkpoint checkpoints/best_checkpoint.pt --data-path {args.output_dir}/test.jsonl")


if __name__ == "__main__":
    main()