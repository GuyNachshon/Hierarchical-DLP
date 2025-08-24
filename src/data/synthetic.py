"""
Synthetic DLP Data Generation

Simplified synthetic data generation for DLP training, consolidated from
the complex agentic system into a clean, maintainable implementation.
"""

from typing import List, Dict, Optional, Any
import json
import random
from pathlib import Path
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass 
class DLPExample:
    """Single DLP training example"""
    channel: str  # email, chat, pr, upload
    user: Dict[str, str]  # role, dept, seniority
    recipients: List[str]
    subject: str
    body: str
    labels: Dict[str, int]  # sensitivity, exposure, context
    spans: List[Dict[str, Any]]  # PII spans


class SyntheticDataGenerator:
    """Generate synthetic DLP training data"""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        
        # Common patterns for data generation
        self.domains = ["legal", "finance", "hr", "security", "casual"]
        self.pii_types = ["EMAIL", "PHONE", "PAN", "SSN", "SECRET_KEY", "DB_URI", "NDA_TERM"]
        
    def generate_example(self) -> DLPExample:
        """Generate a single synthetic example"""
        domain = random.choice(self.domains)
        
        # Basic example structure
        example = DLPExample(
            channel=random.choice(["email", "chat", "pr"]),
            user={
                "role": random.choice(["LEGAL", "FINANCE", "HR", "SECURITY", "EMPLOYEE"]),
                "dept": random.choice(["CORP", "LEGAL", "FIN", "HR", "ENG"]),
                "seniority": random.choice(["JUNIOR", "MID", "SENIOR", "EXEC"])
            },
            recipients=[f"external@{domain}.com"],
            subject=f"Sample {domain} communication",
            body=self._generate_body(domain),
            labels={"sensitivity": random.randint(0, 1), "exposure": random.randint(0, 1), "context": 1},
            spans=[]
        )
        
        return example
    
    def _generate_body(self, domain: str) -> str:
        """Generate realistic body content for domain"""
        templates = {
            "legal": "This contract contains sensitive information about our client relationships.",
            "finance": "Q4 revenue projections show strong growth in our payment processing division.",
            "hr": "Employee performance review scheduled for next week.",
            "security": "Security incident report from last night's system breach.",
            "casual": "Hey team, looking forward to our meeting tomorrow!"
        }
        return templates.get(domain, "Standard business communication content.")
    
    def generate_dataset(self, train_size: int = 1000, val_size: int = 200, test_size: int = 200) -> Dict[str, str]:
        """
        Generate complete dataset with train/val/test splits
        
        Returns:
            Dictionary with paths to generated files
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            "train": train_size,
            "val": val_size, 
            "test": test_size
        }
        
        file_paths = {}
        
        for split_name, size in splits.items():
            examples = [self.generate_example() for _ in range(size)]
            
            file_path = self.output_dir / f"{split_name}.jsonl"
            with open(file_path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example.__dict__) + "\n")
            
            file_paths[split_name] = str(file_path)
        
        return file_paths


class SimpleAgentSystem:
    """Simplified agent system for more realistic data generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.generator = SyntheticDataGenerator("")
        
    def generate_with_llm(self, prompt: str, num_examples: int = 10) -> List[DLPExample]:
        """Generate examples using LLM (simplified from complex batch system)"""
        # Fallback to synthetic generation if no API key
        if not self.api_key:
            return [self.generator.generate_example() for _ in range(num_examples)]
        
        # TODO: Implement actual LLM calls with simple retry logic
        # Much simpler than the original complex batch processing system
        return [self.generator.generate_example() for _ in range(num_examples)]


def generate_dlp_dataset(output_dir: str, train_size: int = 1000, val_size: int = 200, 
                        test_size: int = 200, use_llm: bool = False, api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Main entry point for DLP dataset generation
    
    Args:
        output_dir: Where to save generated data
        train_size, val_size, test_size: Dataset split sizes
        use_llm: Whether to use LLM generation (requires api_key)
        api_key: OpenAI API key for LLM generation
    
    Returns:
        Dictionary with paths to generated files
    """
    if use_llm and api_key:
        agent_system = SimpleAgentSystem(api_key)
        # Use simplified LLM generation
        generator = SyntheticDataGenerator(output_dir)
    else:
        # Use pure synthetic generation
        generator = SyntheticDataGenerator(output_dir)
    
    return generator.generate_dataset(train_size, val_size, test_size)