"""
Simplified state management for resumable data generation.
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Any


class StateManager:
    """Simplified state manager for resumable generation."""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State files
        self.state_file = self.checkpoint_dir / "generation_state.json"
        self.examples_file = self.checkpoint_dir / "completed_examples.jsonl"
        
    def save_state(self, split_name: str, completed_count: int, 
                   pending_requests: List, global_stats: Dict) -> None:
        """Save current generation state."""
        try:
            state = {
                "split_name": split_name,
                "target_size": self._get_target_size(split_name),
                "completed_count": completed_count,
                "pending_requests": [self._serialize_request(req) for req in pending_requests],
                "global_stats": global_stats,
                "timestamp": time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            print(f"💾 State saved: {completed_count} examples completed")
            
        except Exception as e:
            print(f"❌ Failed to save state: {e}")
    
    def load_state(self, split_name: str) -> Optional[Dict]:
        """Load existing generation state."""
        if not self.state_file.exists():
            return None
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            if state.get("split_name") == split_name:
                # Verify against actual completed examples
                actual_count = self.count_completed_examples()
                saved_count = state.get("completed_count", 0)
                
                if actual_count != saved_count:
                    print(f"⚠️  State mismatch: file={actual_count}, saved={saved_count}")
                    state["completed_count"] = actual_count
                
                print(f"📂 Resuming: {actual_count} examples completed")
                return state
            else:
                print(f"🗑️  Different split in state, starting fresh")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load state: {e}")
            return None
    
    def save_example(self, example_dict: Dict) -> None:
        """Save a completed example."""
        try:
            with open(self.examples_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example_dict) + '\\n')
                f.flush()
        except Exception as e:
            print(f"❌ Failed to save example: {e}")
    
    def load_completed_examples(self) -> List[Dict]:
        """Load all completed examples."""
        examples = []
        if self.examples_file.exists():
            try:
                with open(self.examples_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            examples.append(json.loads(line))
            except Exception as e:
                print(f"❌ Failed to load examples: {e}")
        return examples
    
    def count_completed_examples(self) -> int:
        """Count completed examples."""
        if not self.examples_file.exists():
            return 0
            
        try:
            with open(self.examples_file, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
    
    def create_split_files(self) -> None:
        """Create train/val/test split files from completed examples."""
        examples = self.load_completed_examples()
        if not examples:
            print("⚠️  No examples to split")
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create splits
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
                        
                print(f"✅ {split_name}: {len(split_examples)} examples")
                start_idx = end_idx
    
    def clear_state(self) -> None:
        """Clear all persistent state."""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.examples_file.exists():
            self.examples_file.unlink()
        print("🗑️  State cleared")
    
    def _get_target_size(self, split_name: str) -> int:
        """Get target size for a split."""
        if split_name == "train":
            return self.config.train_size
        elif split_name == "val":
            return self.config.val_size
        elif split_name == "test":
            return self.config.test_size
        else:
            return self.config.train_size + self.config.val_size + self.config.test_size
    
    def _serialize_request(self, request) -> Dict:
        """Serialize a GenerationRequest for JSON storage."""
        return {
            "agent_type": request.agent_type,
            "risk_level": request.risk_level,
            "scenario_context": request.scenario_context,
            "target_spans": request.target_spans,
            "conversation_turns": request.conversation_turns,
            "count": request.count
        }