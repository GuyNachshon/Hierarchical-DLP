"""
Fixed Split Batch Coordinator - One batch per split with consistent model usage.

Key fixes:
- Each split becomes exactly one batch (unless >140K entries)
- Each batch uses exactly one model for all requests
- Maintains input file persistence and recovery capabilities
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
    """Plan for generating a specific split - now as single batch."""
    split_name: str
    target_size: int
    requests: List[Any]  # Generation requests
    max_batch_size: int = 140000  # Maximum entries per batch
    estimated_batches: int = 0
    
    def __post_init__(self):
        # Calculate how many batches needed based on max size
        self.estimated_batches = max(1, (len(self.requests) + self.max_batch_size - 1) // self.max_batch_size)


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


class FixedSplitBatchCoordinator:
    """Fixed coordinator - one batch per split with consistent models."""
    
    def __init__(self, session_manager: RunSessionManager, 
                 batch_tracker: EnhancedBatchTracker,
                 batch_processor: BatchProcessor):
        self.session_manager = session_manager
        self.batch_tracker = batch_tracker
        self.batch_processor = batch_processor
        
        # Split progress tracking
        self.split_progress: Dict[str, SplitProgress] = {}
        
        # Configuration
        self.max_entries_per_batch = 140000  # Maximum entries per batch
        self.batch_retry_count = 2
    
    async def generate_split_dataset(self, split_plan: SplitGenerationPlan) -> List[Dict]:
        """Generate dataset for a single split as one batch (unless >140K)."""
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
        remaining_requests = split_plan.requests[:remaining_needed]
        
        print(f"📊 {split_name}: {len(completed_examples)} existing + {remaining_needed} needed = {split_plan.target_size} total")
        
        # Check if we need to split into multiple batches (only if >140K)
        if len(remaining_requests) <= self.max_entries_per_batch:
            # Single batch for the entire split
            print(f"📦 {split_name}: Processing as single batch ({len(remaining_requests)} requests)")
            
            new_examples = await self._process_single_split_batch(
                split_name, remaining_requests, batch_num=1
            )
        else:
            # Multiple batches needed (rare case for very large splits)
            print(f"📦 {split_name}: Large split requires {split_plan.estimated_batches} batches")
            
            new_examples = []
            for batch_idx in range(split_plan.estimated_batches):
                start_idx = batch_idx * self.max_entries_per_batch
                end_idx = min(start_idx + self.max_entries_per_batch, len(remaining_requests))
                batch_requests = remaining_requests[start_idx:end_idx]
                
                print(f"🔄 {split_name}: Processing batch {batch_idx + 1}/{split_plan.estimated_batches} ({len(batch_requests)} requests)")
                
                batch_examples = await self._process_single_split_batch(
                    split_name, batch_requests, batch_num=batch_idx + 1
                )
                new_examples.extend(batch_examples)
        
        # Save all examples
        for example in new_examples:
            await self._save_split_example(split_name, example)
        
        # Mark split as completed
        self.split_progress[split_name].status = "completed"
        self.split_progress[split_name].completed_examples = len(completed_examples) + len(new_examples)
        
        all_examples = completed_examples + new_examples
        print(f"🎉 {split_name} split generation completed! ({len(all_examples)} examples)")
        
        return all_examples
    
    async def _process_single_split_batch(self, split_name: str, requests: List[Any], batch_num: int) -> List[Dict]:
        """Process a single batch for a split using consistent model."""
        # Choose ONE model for the entire batch
        provider, model = await self._choose_consistent_model_for_batch(requests)
        
        print(f"🤖 {split_name} batch {batch_num}: Using {provider}/{model} for all {len(requests)} requests")
        
        # Build prompts for each request
        request_pairs = []
        for req in requests:
            system_prompt, user_prompt = self._build_prompts_for_request(req)
            # Attach for transparency/debugging
            try:
                setattr(req, 'system_prompt', system_prompt)
                setattr(req, 'user_prompt', user_prompt)
            except Exception:
                pass
            request_pairs.append((system_prompt, user_prompt))

        # Save batch input with model info and explicit prompts
        input_file = await self._save_batch_input(
            split_name, requests, batch_num, provider, model, request_pairs
        )
        
        # Register batch with tracker
        batch_id = f"{split_name}_batch_{batch_num:03d}_{provider}_{model}"
        
        batch_metadata = self.batch_tracker.create_batch(
            batch_id=batch_id,
            provider=provider,
            model=model,
            request_count=len(requests),
            split_name=split_name,
            request_indices=list(range(len(requests))),
            input_file=str(input_file)
        )
        
        try:
            # Process through batch processor with fixed model
            responses = await self.batch_processor.process_requests_with_fixed_model(
                request_pairs, provider, model
            )
            
            # Convert responses to examples
            examples = []
            for i, (req, response) in enumerate(zip(requests, responses)):
                if response:
                    example = self._convert_response_to_example(req, response, split_name, batch_num, i, provider, model)
                    if example:
                        examples.append(example)
            
            # Mark batch as completed
            self.batch_tracker.complete_batch(batch_id, success=True)
            
            print(f"✅ {split_name} batch {batch_num}: {len(examples)}/{len(requests)} examples generated using {provider}/{model}")
            return examples
            
        except Exception as e:
            print(f"❌ Batch processing failed for {split_name} batch {batch_num}: {e}")
            self.batch_tracker.complete_batch(batch_id, success=False, error_message=str(e))
            raise
    
    async def _choose_consistent_model_for_batch(self, requests: List[Any]) -> Tuple[str, str]:
        """Choose ONE model for the entire batch based on request characteristics."""
        # Analyze requests to determine best model
        # For now, use a simple strategy but this could be made smarter
        
        # Count different agent types in requests
        agent_types = {}
        for req in requests:
            agent_type = getattr(req, 'agent_type', 'unknown')
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Choose model based on dominant agent type
        dominant_agent = max(agent_types.items(), key=lambda x: x[1])[0] if agent_types else 'unknown'
        
        # Model selection strategy based on agent type
        if dominant_agent in ['legal', 'finance', 'security']:
            # Complex reasoning tasks - use more capable model
            preferred_provider = "openai"
        elif dominant_agent in ['casual', 'clean_business']:
            # Simpler tasks - use faster model
            preferred_provider = "openai"
        else:
            # Default to anthropic for robustness
            preferred_provider = "anthropic"
        
        # Get the actual model from batch processor
        provider, model = self.batch_processor.client_manager.choose_model(preferred_provider)
        
        return provider, model
    
    def _convert_response_to_example(self, request: Any, response: str, 
                                   split_name: str, batch_num: int, req_index: int,
                                   provider: str, model: str) -> Optional[Dict]:
        """Convert a response to a DLP example format with model info."""
        try:
            # Short-circuit placeholder markers (submit-only mode)
            if isinstance(response, str) and response.startswith("BATCH_"):
                return None

            # Parse response if it's JSON
            if response.strip().startswith('{'):
                example_data = json.loads(response)
            else:
                # Create basic example structure with cleanup and synthesized subject
                clean_body = self._clean_text_artifacts(response)
                example_data = {
                    "channel": "email",
                    "subject": self._synthesize_subject_from_body(clean_body),
                    "body": clean_body,
                    "labels": {"sensitivity": 0, "exposure": 0, "context": 1},
                    "spans": []
                }
            
            # Add comprehensive metadata
            example_data["_metadata"] = {
                "split": split_name,
                "batch_num": batch_num,
                "request_index": req_index,
                "agent_type": getattr(request, 'agent_type', 'unknown'),
                "provider": provider,
                "model": model,
                "batch_id": f"{split_name}_batch_{batch_num:03d}_{provider}_{model}"
            }
            
            return example_data
            
        except Exception as e:
            print(f"⚠️  Failed to convert response to example: {e}")
            return None

    def _clean_text_artifacts(self, text: str) -> str:
        """Remove common artifacts like 'Turn N' and code fences."""
        import re
        text = re.sub(r"(?i)\bturn\s*\d+\s*:?\s*", "", text)
        text = text.replace("```", "").strip()
        return text

    def _synthesize_subject_from_body(self, body: str) -> str:
        """Derive a subject from the first sentence or line of body."""
        import re
        if not body:
            return "Generated Email"
        first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), body.strip())
        m = re.match(r"(.{1,80}?)([\.!?]|$)", first_line)
        subj = m.group(1).strip() if m else first_line[:80]
        subj = re.sub(r"(?i)\bturn\s*\d+\b", "", subj).strip()
        return subj or "Generated Email"
    
    def _build_prompts_for_request(self, req: Any) -> Tuple[str, str]:
        """Build (system, user) prompts from a GenerationRequest-like object."""
        agent_type = getattr(req, 'agent_type', 'general')
        risk_level = getattr(req, 'risk_level', 'medium_risk')
        scenario_context = getattr(req, 'scenario_context', 'general')
        target_spans = getattr(req, 'target_spans', [])
        conversation_turns = getattr(req, 'conversation_turns', 1)

        system_prompt = (
            f"You are a {agent_type} domain expert generating {risk_level} communication."
        )
        user_prompt = (
            f"Scenario: {scenario_context}\n"
            f"Target spans: {target_spans}\n"
            f"Turns: {conversation_turns}"
        )
        return system_prompt, user_prompt

    async def _save_batch_input(self, split_name: str, requests: List[Any], batch_num: int,
                               provider: str, model: str, request_pairs: List[Tuple[str, str]]) -> Path:
        """Save batch input requests to file with model info and explicit prompts."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            raise RuntimeError("No active session directory")
        
        input_dir = session_dir / "batch_inputs"
        input_dir.mkdir(exist_ok=True)
        
        input_file = input_dir / f"{split_name}_batch_{batch_num:03d}_{provider}_{model}_input.jsonl"
        
        # Create comprehensive batch input record
        batch_record = {
            "split_name": split_name,
            "batch_num": batch_num,
            "provider": provider,
            "model": model,
            "request_count": len(requests),
            "timestamp": asyncio.get_event_loop().time(),
            "requests": []
        }
        
        # Convert requests to serializable format
        for i, req in enumerate(requests):
            if hasattr(req, 'to_dict'):
                req_data = req.to_dict()
            elif hasattr(req, '__dict__'):
                req_data = req.__dict__
            else:
                req_data = {"content": str(req)}
            
            req_data["_index"] = i
            # Attach explicit prompts that will be sent
            try:
                system_prompt, user_prompt = request_pairs[i]
                req_data["system_prompt"] = system_prompt
                req_data["user_prompt"] = user_prompt
            except Exception:
                pass
            batch_record["requests"].append(req_data)
        
        # Save as single JSON record for the batch
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(batch_record, indent=2))
        
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
        """Generate datasets for all splits - each as single batch."""
        print(f"🚀 Starting fixed generation for {len(split_plans)} splits (one batch each)")
        results: Dict[str, List[Dict]] = {}
        
        submit_parallel = getattr(self.batch_processor.config, 'submit_splits_concurrently', True)
        if submit_parallel:
            print("🔀 Submitting splits in parallel (non-blocking between them)")
            tasks = [asyncio.create_task(self.generate_split_dataset(plan)) for plan in split_plans]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for plan, result in zip(split_plans, gathered):
                if isinstance(result, Exception):
                    print(f"❌ {plan.split_name} split failed: {result}")
                    results[plan.split_name] = []
                else:
                    results[plan.split_name] = result
                    print(f"✅ {plan.split_name} completed: {len(result)} examples")
        else:
            for plan in split_plans:
                try:
                    print(f"📊 Processing {plan.split_name} split...")
                    split_examples = await self.generate_split_dataset(plan)
                    results[plan.split_name] = split_examples
                    print(f"✅ {plan.split_name} completed: {len(split_examples)} examples")
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"❌ {plan.split_name} split failed: {e}")
                    results[plan.split_name] = []
        
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
                
                # Also create a metadata file showing which model was used
                metadata_file = output_dir / f"{split_name}_metadata.json"
                if examples:
                    first_example_meta = examples[0].get("_metadata", {})
                    split_metadata = {
                        "split_name": split_name,
                        "total_examples": len(examples),
                        "provider": first_example_meta.get("provider", "unknown"),
                        "model": first_example_meta.get("model", "unknown"),
                        "batch_count": len(set(ex.get("_metadata", {}).get("batch_id") for ex in examples)),
                        "agent_distribution": self._calculate_agent_distribution(examples)
                    }
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(split_metadata, f, indent=2)
                
                print(f"💾 Final {split_name}.jsonl: {len(examples)} examples (model: {first_example_meta.get('provider')}/{first_example_meta.get('model')})")
            else:
                print(f"⚠️  No examples generated for {split_name} split")
    
    def _calculate_agent_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of examples by agent type."""
        agent_counts = {}
        for example in examples:
            agent_type = example.get("_metadata", {}).get("agent_type", "unknown")
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        return agent_counts
    
    def cleanup_split(self, split_name: str):
        """Clean up progress tracking for a split."""
        if split_name in self.split_progress:
            del self.split_progress[split_name]
        
        # Clean up split-specific batches
        active_batches = self.batch_tracker.get_active_batches_for_split(split_name)
        for batch_id in active_batches:
            self.batch_tracker.complete_batch(batch_id, success=False, error_message="Split cleanup")
        
        print(f"🗑️  Cleaned up {split_name} split progress")
