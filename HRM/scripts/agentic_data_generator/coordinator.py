"""
Main coordinator for agentic data generation with simplified architecture.
"""

import os
import time
import asyncio
import json
from typing import List, Optional, Dict
from pathlib import Path

from .config import AgenticConfig
from .state_manager import StateManager
from .agents import (
    ManagerAgent, LegalAgent, FinanceAgent, HRAgent, SecurityAgent,
    CasualAgent, CleanBusinessAgent, ObfuscationSpecialist, ConversationalAgent
)
from .batch.batch_processor import BatchProcessor
from .data.validators import QualityValidator
from .data.converters import DLPFormatConverter


class AgenticDataGenerator:
    """Simplified main coordinator for agentic data generation."""
    
    def __init__(self, config: AgenticConfig):
        self.config = config
        self.state_manager = StateManager(config) if config.enable_resume else None
        self.batch_processor = BatchProcessor(config)
        self.quality_validator = QualityValidator()
        self.dlp_converter = DLPFormatConverter()
        
        # Initialize agents
        self.manager = ManagerAgent(config)
        self.agents = {
            "legal": LegalAgent(config),
            "finance": FinanceAgent(config),
            "hr": HRAgent(config),
            "security": SecurityAgent(config),
            "casual": CasualAgent(config),
            "clean_business": CleanBusinessAgent(config),
            "obfuscation": ObfuscationSpecialist(config)
        }
        self.conversational_agent = ConversationalAgent(config)
        
        # Statistics
        self.global_stats = {
            "total_requested": 0,
            "total_generated": 0,
            "quality_rejections": 0,
            "agent_stats": {name: {"requested": 0, "generated": 0} for name in self.agents.keys()}
        }
    
    async def generate_dataset(self) -> None:
        """Generate complete dataset using simplified approach."""
        print("🚀 Starting Agentic Data Generator...")
        print(f"📊 Target: {self._get_total_target()} examples")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Generate all examples in one go
        all_examples = await self._generate_all_examples()
        
        # Save examples and create split files
        self._save_dataset(all_examples)
        
        print(f"✅ Dataset generation complete!")
        print(f"📁 Output directory: {self.config.output_dir}")
    
    async def _generate_all_examples(self) -> List[Dict]:
        """Generate all examples for the dataset."""
        target_size = self._get_total_target()
        
        # Try to resume from checkpoint
        completed_examples = []
        if self.state_manager:
            completed_examples = self.state_manager.load_completed_examples()
            if len(completed_examples) >= target_size:
                print(f"✅ Found {len(completed_examples)} completed examples")
                return completed_examples[:target_size]
        
        # Generate remaining examples
        remaining_needed = target_size - len(completed_examples)
        print(f"🔄 Generating {remaining_needed} new examples...")
        
        # Create generation plan
        requests = await self.manager.create_generation_plan("all", remaining_needed)
        
        # Process requests in batches
        new_examples = []
        batch_size = 50  # Process in smaller batches for better progress tracking
        
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(requests) + batch_size - 1)//batch_size}")
            
            batch_examples = await self._process_request_batch(batch_requests)
            
            # Save examples immediately
            for example in batch_examples:
                if self.state_manager:
                    self.state_manager.save_example(example)
                new_examples.append(example)
            
            print(f"✅ Batch complete: {len(batch_examples)} examples generated")
        
        return completed_examples + new_examples
    
    async def _process_request_batch(self, requests: List) -> List[Dict]:
        """Process a batch of generation requests."""
        examples = []
        
        # Group requests by agent type for efficiency
        agent_batches = {}
        for request in requests:
            agent_type = request.agent_type
            if agent_type not in agent_batches:
                agent_batches[agent_type] = []
            agent_batches[agent_type].append(request)
        
        # Process each agent type
        for agent_type, agent_requests in agent_batches.items():
            if agent_type == "conversational":
                # Handle multi-turn conversations
                for request in agent_requests:
                    thread_examples = await self.conversational_agent.generate_conversation_thread(request)
                    for example in thread_examples:
                        if self._validate_and_convert_example(example, request):
                            dlp_example = self.dlp_converter.convert_to_dlp_format(example)
                            if dlp_example:
                                examples.append(dlp_example)
            else:
                # Handle single examples
                agent = self.agents.get(agent_type)
                if agent:
                    for request in agent_requests:
                        for _ in range(request.count):
                            example = await agent.generate_example(request)
                            if self._validate_and_convert_example(example, request):
                                dlp_example = self.dlp_converter.convert_to_dlp_format(example)
                                if dlp_example:
                                    examples.append(dlp_example)
        
        return examples
    
    def _validate_and_convert_example(self, example, request) -> bool:
        """Validate example quality and update stats."""
        if not example:
            self.global_stats["quality_rejections"] += 1
            return False
        
        # Quality validation
        if not self.quality_validator.validate_example(example, self.config.min_quality_score):
            self.global_stats["quality_rejections"] += 1
            return False
        
        # Agent-specific validation
        if not self.quality_validator.validate_agent_specific(example, request.agent_type):
            self.global_stats["quality_rejections"] += 1
            return False
        
        # Update stats
        agent_type = request.agent_type
        if agent_type in self.global_stats["agent_stats"]:
            self.global_stats["agent_stats"][agent_type]["generated"] += 1
        
        self.global_stats["total_generated"] += 1
        return True
    
    def _save_dataset(self, examples: List[Dict]) -> None:
        """Save examples and create train/val/test splits."""
        if not examples:
            print("⚠️  No examples to save")
            return
        
        # Save all examples to checkpoint
        if self.state_manager:
            self.state_manager.examples_file.unlink(missing_ok=True)  # Clear old file
            for example in examples:
                self.state_manager.save_example(example)
        
        # Create split files
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = [
            ("train", self.config.train_size),
            ("val", self.config.val_size),
            ("test", self.config.test_size)
        ]
        
        start_idx = 0
        for split_name, split_size in splits:
            end_idx = start_idx + split_size
            split_examples = examples[start_idx:min(end_idx, len(examples))]
            
            if split_examples:
                split_file = output_dir / f"{split_name}.jsonl"
                with open(split_file, 'w', encoding='utf-8') as f:
                    for example in split_examples:
                        f.write(json.dumps(example) + '\\n')
                
                print(f"✅ {split_name}: {len(split_examples)} examples saved")
                start_idx = end_idx
        
        # Save statistics
        stats_file = output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "config": {
                    "train_size": self.config.train_size,
                    "val_size": self.config.val_size,
                    "test_size": self.config.test_size,
                    "agent_distribution": self.config.agent_distribution,
                    "risk_distribution": self.config.risk_distribution
                },
                "stats": self.global_stats,
                "total_examples": len(examples)
            }, f, indent=2)
        
        print(f"📊 Statistics saved to {stats_file}")
    
    def _get_total_target(self) -> int:
        """Get total target size for dataset."""
        return self.config.train_size + self.config.val_size + self.config.test_size