"""
Split Batch Coordinator - Orchestrates per-split batch processing.

Handles independent generation and processing of train/val/test splits,
with each split having its own batch lifecycle and recovery capabilities.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .run_session_manager import RunSessionManager
from .enhanced_batch_tracker import EnhancedBatchTracker, BatchStatus
from .batch_processor import BatchProcessor


@dataclass
class SplitGenerationPlan:
    """Plan for generating a specific split."""
    split_name: str
    target_size: int
    requests: List[Any]  # Generation requests
    batch_size: int = 50
    estimated_batches: int = 0
    
    def __post_init__(self):
        if self.estimated_batches == 0:
            self.estimated_batches = (len(self.requests) + self.batch_size - 1) // self.batch_size


@dataclass
class SplitProgress:
    """Progress tracking for a split."""
    split_name: str
    target_size: int
    completed_examples: int = 0
    active_batches: int = 0
    failed_batches: int = 0
    completed_batches: int = 0
    status: str = "pending"  # pending, generating, completing, completed, failed


class SplitBatchCoordinator:
    """Coordinates batch processing for individual splits."""
    
    def __init__(self, session_manager: RunSessionManager, 
                 batch_tracker: EnhancedBatchTracker,
                 batch_processor: BatchProcessor):
        self.session_manager = session_manager
        self.batch_tracker = batch_tracker
        self.batch_processor = batch_processor
        
        # Split progress tracking
        self.split_progress: Dict[str, SplitProgress] = {}
        
        # Configuration
        self.max_concurrent_batches_per_split = 3
        self.batch_retry_count = 2
    
    async def generate_split_dataset(self, split_plan: SplitGenerationPlan) -> List[Dict]:
        """Generate dataset for a single split."""
        split_name = split_plan.split_name
        
        print(f"🎯 Starting generation for {split_name} split ({split_plan.target_size} examples)")
        
        # Initialize progress tracking
        self.split_progress[split_name] = SplitProgress(
            split_name=split_name,
            target_size=split_plan.target_size,
            status="generating"
        )
        
        # Try to resume from existing progress
        completed_examples = await self._load_split_progress(split_name)
        if len(completed_examples) >= split_plan.target_size:
            print(f"✅ {split_name} split already completed ({len(completed_examples)} examples)")
            self.split_progress[split_name].status = "completed"
            return completed_examples[:split_plan.target_size]
        
        # Calculate remaining work
        remaining_needed = split_plan.target_size - len(completed_examples)
        remaining_requests = split_plan.requests[:remaining_needed]  # Adjust requests to remaining need
        
        print(f"📊 {split_name}: {len(completed_examples)} existing + {remaining_needed} needed = {split_plan.target_size} total")
        
        # Process requests in batches
        new_examples = []
        
        try:
            for batch_start in range(0, len(remaining_requests), split_plan.batch_size):
                batch_requests = remaining_requests[batch_start:batch_start + split_plan.batch_size]
                batch_num = (batch_start // split_plan.batch_size) + 1
                total_batches = (len(remaining_requests) + split_plan.batch_size - 1) // split_plan.batch_size
                
                print(f"🔄 {split_name} batch {batch_num}/{total_batches} ({len(batch_requests)} requests)")
                
                # Wait if we have too many concurrent batches
                await self._wait_for_batch_slot(split_name)
                
                # Process batch
                batch_examples = await self._process_split_batch(split_name, batch_requests, batch_num)
                
                # Save examples immediately
                for example in batch_examples:
                    await self._save_split_example(split_name, example)
                    new_examples.append(example)
                
                # Update progress
                self.split_progress[split_name].completed_examples = len(completed_examples) + len(new_examples)
                
                print(f"✅ {split_name} batch {batch_num} complete: {len(batch_examples)} examples")
            
            # Mark split as completed
            self.split_progress[split_name].status = "completed"
            all_examples = completed_examples + new_examples
            
            print(f"🎉 {split_name} split generation completed! ({len(all_examples)} examples)")
            return all_examples
            
        except Exception as e:
            print(f"❌ {split_name} split generation failed: {e}")
            self.split_progress[split_name].status = "failed"
            raise
    
    async def _wait_for_batch_slot(self, split_name: str):
        """Wait for available batch slot for the split."""
        while True:
            active_batches = self.batch_tracker.get_active_batches_for_split(split_name)
            if len(active_batches) < self.max_concurrent_batches_per_split:
                break
            
            print(f"⏳ {split_name}: Waiting for batch slot ({len(active_batches)}/{self.max_concurrent_batches_per_split} active)")
            await asyncio.sleep(10)  # Wait 10 seconds before checking again
    
    async def _process_split_batch(self, split_name: str, requests: List[Any], batch_num: int) -> List[Dict]:
        """Process a batch of requests for a specific split."""
        # Save batch input
        input_file = await self._save_batch_input(split_name, requests, batch_num)
        
        # Convert requests to (system, prompt) pairs
        request_pairs = []
        for req in requests:
            # This would need to be implemented based on your request structure
            system_prompt = getattr(req, 'system_prompt', 'You are a helpful assistant.')
            user_prompt = getattr(req, 'user_prompt', str(req))
            request_pairs.append((system_prompt, user_prompt))
        
        # Process through batch processor
        try:
            responses = await self.batch_processor.process_requests(request_pairs)
            
            # Convert responses to examples
            examples = []
            for i, (req, response) in enumerate(zip(requests, responses)):
                if response:
                    # This would need to be implemented based on your example format
                    example = self._convert_response_to_example(req, response, split_name, batch_num, i)
                    if example:
                        examples.append(example)
            
            return examples
            
        except Exception as e:
            print(f"❌ Batch processing failed for {split_name} batch {batch_num}: {e}")
            raise
    
    def _convert_response_to_example(self, request: Any, response: str, 
                                   split_name: str, batch_num: int, req_index: int) -> Optional[Dict]:
        """Convert a response to a DLP example format."""
        # This is a placeholder - would need to be implemented based on your specific format
        try:
            # Parse response if it's JSON
            if response.strip().startswith('{'):
                example_data = json.loads(response)
            else:
                # Create basic example structure
                example_data = {
                    "channel": "email",
                    "subject": "Generated Email",
                    "body": response,
                    "labels": {"sensitivity": 0, "exposure": 0, "context": 1},
                    "spans": []
                }
            
            # Add metadata
            example_data["_metadata"] = {
                "split": split_name,
                "batch_num": batch_num,
                "request_index": req_index,
                "agent_type": getattr(request, 'agent_type', 'unknown')
            }
            
            return example_data
            
        except Exception as e:
            print(f"⚠️  Failed to convert response to example: {e}")
            return None
    
    async def _save_batch_input(self, split_name: str, requests: List[Any], batch_num: int) -> Path:
        """Save batch input requests to file."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            raise RuntimeError("No active session directory")
        
        input_dir = session_dir / "batch_inputs"
        input_dir.mkdir(exist_ok=True)
        
        input_file = input_dir / f"{split_name}_batch_{batch_num:03d}_input.jsonl"
        
        # Convert requests to serializable format
        serializable_requests = []
        for req in requests:
            if hasattr(req, 'to_dict'):
                serializable_requests.append(req.to_dict())
            elif hasattr(req, '__dict__'):
                serializable_requests.append(req.__dict__)
            else:
                serializable_requests.append(str(req))
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for req in serializable_requests:
                f.write(json.dumps(req) + '\n')
        
        print(f"💾 Saved batch input: {input_file}")
        return input_file
    
    async def _save_split_example(self, split_name: str, example: Dict):
        """Save a completed example for the split."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            raise RuntimeError("No active session directory")
        
        output_dir = session_dir / "split_outputs"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{split_name}_examples.jsonl"
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example) + '\n')
    
    async def _load_split_progress(self, split_name: str) -> List[Dict]:
        """Load existing progress for a split."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            return []
        
        output_file = session_dir / "split_outputs" / f"{split_name}_examples.jsonl"
        
        if not output_file.exists():
            return []
        
        examples = []
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
            
            print(f"📂 Loaded {len(examples)} existing examples for {split_name}")
            return examples
            
        except Exception as e:
            print(f"⚠️  Failed to load split progress for {split_name}: {e}")
            return []
    
    def get_split_progress(self, split_name: str) -> Optional[SplitProgress]:
        """Get current progress for a split."""
        return self.split_progress.get(split_name)
    
    def get_all_progress(self) -> Dict[str, SplitProgress]:
        """Get progress for all splits."""
        return self.split_progress.copy()
    
    async def generate_all_splits(self, split_plans: List[SplitGenerationPlan]) -> Dict[str, List[Dict]]:
        """Generate datasets for all splits concurrently."""
        print(f"🚀 Starting generation for {len(split_plans)} splits")
        
        # Start all splits concurrently
        tasks = {
            plan.split_name: asyncio.create_task(self.generate_split_dataset(plan))
            for plan in split_plans
        }
        
        results = {}
        
        # Wait for each split to complete
        for split_name, task in tasks.items():
            try:
                results[split_name] = await task
                print(f"✅ {split_name} split completed")
            except Exception as e:
                print(f"❌ {split_name} split failed: {e}")
                results[split_name] = []
        
        return results
    
    async def create_final_split_files(self, results: Dict[str, List[Dict]], output_dir: Path):
        """Create final train/val/test.jsonl files from split results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, examples in results.items():
            if examples:
                output_file = output_dir / f"{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in examples:
                        # Remove metadata before saving final file
                        final_example = {k: v for k, v in example.items() if not k.startswith('_')}
                        f.write(json.dumps(final_example) + '\n')
                
                print(f"💾 Final {split_name}.jsonl: {len(examples)} examples")
            else:
                print(f"⚠️  No examples generated for {split_name} split")
    
    def cleanup_split(self, split_name: str):
        """Clean up progress tracking for a split."""
        if split_name in self.split_progress:
            del self.split_progress[split_name]
        
        # Clean up split-specific batches
        active_batches = self.batch_tracker.get_active_batches_for_split(split_name)
        for batch_id in active_batches:
            self.batch_tracker.complete_batch(batch_id, success=False, error_message="Split cleanup")
        
        print(f"🗑️  Cleaned up {split_name} split progress")