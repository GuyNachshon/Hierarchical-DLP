"""
Enhanced Batch Tracker with run session awareness and split support.

Extends the original BatchTracker to provide:
- Run session isolation
- Split-specific batch tracking  
- Safe cleanup operations
- Enhanced recovery capabilities
"""

import json
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .run_session_manager import RunSessionManager


class BatchStatus(Enum):
    """Batch status enumeration"""
    CREATED = "created"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class EnhancedBatchMetadata:
    """Enhanced metadata for tracking a batch with run session awareness."""
    batch_id: str
    provider: str  # "openai" or "anthropic"
    model: str
    request_count: int
    created_at: float
    timeout_at: float
    
    # Enhanced fields
    run_session_id: str  # Which run session this batch belongs to
    split_name: str      # Which split (train/val/test) this batch is for
    input_file: Optional[str] = None  # Path to the input JSONL file
    
    # Original fields
    status: str = BatchStatus.CREATED.value
    request_indices: List[int] = None  # Original indices for result mapping
    results_retrieved: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.request_indices is None:
            self.request_indices = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedBatchMetadata':
        """Create from dictionary."""
        return cls(**data)


class EnhancedBatchTracker:
    """Enhanced batch tracker with run session awareness and split support."""
    
    def __init__(self, session_manager: RunSessionManager, batch_timeout: int = 86400):
        self.session_manager = session_manager
        self.batch_timeout = batch_timeout
        
        # In-memory state (organized by session)
        self.active_batches: Dict[str, Dict[str, EnhancedBatchMetadata]] = {}  # session_id -> batch_id -> metadata
        self.completed_batches: Dict[str, Dict[str, EnhancedBatchMetadata]] = {}
        
        # Load existing state
        self._load_state()
    
    def _get_session_batch_dir(self, session_id: str) -> Path:
        """Get the batch directory for a specific session."""
        session_dir = self.session_manager.runs_dir / session_id
        batch_dir = session_dir / "checkpoints" / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir
    
    def _load_state(self):
        """Load existing batch state from all sessions."""
        for session in self.session_manager.list_all_sessions():
            session_id = session.session_id
            batch_dir = self._get_session_batch_dir(session_id)
            
            # Load active batches for this session
            active_file = batch_dir / "active_batches.json"
            if active_file.exists():
                try:
                    with open(active_file, 'r') as f:
                        data = json.load(f)
                        session_active = {
                            batch_id: EnhancedBatchMetadata.from_dict(batch_data)
                            for batch_id, batch_data in data.items()
                        }
                        if session_active:
                            self.active_batches[session_id] = session_active
                except Exception as e:
                    print(f"⚠️  Failed to load active batches for session {session_id}: {e}")
            
            # Load completed batches for this session
            completed_file = batch_dir / "completed_batches.json"
            if completed_file.exists():
                try:
                    with open(completed_file, 'r') as f:
                        data = json.load(f)
                        session_completed = {
                            batch_id: EnhancedBatchMetadata.from_dict(batch_data)
                            for batch_id, batch_data in data.items()
                        }
                        if session_completed:
                            self.completed_batches[session_id] = session_completed
                except Exception as e:
                    print(f"⚠️  Failed to load completed batches for session {session_id}: {e}")
    
    def _save_session_state(self, session_id: str):
        """Save state for a specific session."""
        batch_dir = self._get_session_batch_dir(session_id)
        
        try:
            # Save active batches
            active_file = batch_dir / "active_batches.json"
            session_active = self.active_batches.get(session_id, {})
            with open(active_file, 'w') as f:
                json.dump(
                    {batch_id: metadata.to_dict() for batch_id, metadata in session_active.items()},
                    f, indent=2
                )
            
            # Save completed batches (keep last 100)
            completed_file = batch_dir / "completed_batches.json"
            session_completed = self.completed_batches.get(session_id, {})
            recent_completed = dict(list(session_completed.items())[-100:])
            with open(completed_file, 'w') as f:
                json.dump(
                    {batch_id: metadata.to_dict() for batch_id, metadata in recent_completed.items()},
                    f, indent=2
                )
                
        except Exception as e:
            print(f"❌ Failed to save batch state for session {session_id}: {e}")
    
    def create_batch(self, batch_id: str, provider: str, model: str, 
                    request_count: int, split_name: str, 
                    request_indices: List[int], input_file: Optional[str] = None) -> EnhancedBatchMetadata:
        """Register a new batch for tracking."""
        current_session = self.session_manager.get_current_session()
        if not current_session:
            raise RuntimeError("No active run session")
        
        session_id = current_session.session_id
        current_time = time.time()
        
        metadata = EnhancedBatchMetadata(
            batch_id=batch_id,
            provider=provider,
            model=model,
            request_count=request_count,
            created_at=current_time,
            timeout_at=current_time + self.batch_timeout,
            run_session_id=session_id,
            split_name=split_name,
            input_file=input_file,
            request_indices=request_indices.copy()
        )
        
        # Initialize session if needed
        if session_id not in self.active_batches:
            self.active_batches[session_id] = {}
        
        self.active_batches[session_id][batch_id] = metadata
        self._save_session_state(session_id)
        
        print(f"📝 Registered batch {batch_id[:8]}... for {split_name} split ({provider}/{model}, {request_count} requests)")
        return metadata
    
    def update_batch_status(self, batch_id: str, status: BatchStatus, error_message: Optional[str] = None):
        """Update batch status."""
        # Find the batch across all sessions
        for session_id, session_batches in self.active_batches.items():
            if batch_id in session_batches:
                session_batches[batch_id].status = status.value
                if error_message:
                    session_batches[batch_id].error_message = error_message
                self._save_session_state(session_id)
                return
    
    def complete_batch(self, batch_id: str, success: bool = True, error_message: Optional[str] = None):
        """Mark batch as completed and move to completed list."""
        # Find the batch across all sessions
        for session_id, session_batches in self.active_batches.items():
            if batch_id in session_batches:
                metadata = session_batches[batch_id]
                metadata.status = BatchStatus.COMPLETED.value if success else BatchStatus.FAILED.value
                metadata.results_retrieved = success
                if error_message:
                    metadata.error_message = error_message
                
                # Move to completed
                if session_id not in self.completed_batches:
                    self.completed_batches[session_id] = {}
                self.completed_batches[session_id][batch_id] = metadata
                del session_batches[batch_id]
                
                self._save_session_state(session_id)
                
                status_emoji = "✅" if success else "❌"
                print(f"{status_emoji} Batch {batch_id[:8]}... for {metadata.split_name} split completed ({'success' if success else 'failed'})")
                return
    
    def get_active_batches_for_session(self, session_id: Optional[str] = None) -> Dict[str, EnhancedBatchMetadata]:
        """Get active batches for a specific session (or current session)."""
        if session_id is None:
            current_session = self.session_manager.get_current_session()
            if not current_session:
                return {}
            session_id = current_session.session_id
        
        return self.active_batches.get(session_id, {}).copy()
    
    def get_active_batches_for_split(self, split_name: str, session_id: Optional[str] = None) -> Dict[str, EnhancedBatchMetadata]:
        """Get active batches for a specific split in a session."""
        session_batches = self.get_active_batches_for_session(session_id)
        return {
            batch_id: metadata for batch_id, metadata in session_batches.items()
            if metadata.split_name == split_name
        }
    
    def get_all_active_batches(self) -> Dict[str, EnhancedBatchMetadata]:
        """Get all active batches across all sessions."""
        all_batches = {}
        for session_batches in self.active_batches.values():
            all_batches.update(session_batches)
        return all_batches
    
    def get_timed_out_batches(self, session_id: Optional[str] = None) -> List[EnhancedBatchMetadata]:
        """Get batches that have exceeded timeout."""
        current_time = time.time()
        timed_out = []
        
        batches = self.get_active_batches_for_session(session_id) if session_id else self.get_all_active_batches()
        
        for metadata in batches.values():
            if (current_time > metadata.timeout_at and 
                metadata.status not in [BatchStatus.COMPLETED.value, BatchStatus.FAILED.value]):
                timed_out.append(metadata)
        
        return timed_out
    
    def cleanup_timed_out_batches_for_session(self, session_id: Optional[str] = None) -> List[str]:
        """Mark timed-out batches as expired for a specific session."""
        timed_out = self.get_timed_out_batches(session_id)
        expired_ids = []
        
        for metadata in timed_out:
            print(f"⏰ Batch {metadata.batch_id[:8]}... ({metadata.split_name}) timed out after {self.batch_timeout}s")
            self.complete_batch(metadata.batch_id, success=False, error_message="Batch timed out")
            expired_ids.append(metadata.batch_id)
        
        return expired_ids
    
    def get_batch_count_for_split(self, split_name: str, provider: str, model: str, session_id: Optional[str] = None) -> int:
        """Get count of active batches for a specific split and model."""
        split_batches = self.get_active_batches_for_split(split_name, session_id)
        return sum(
            1 for metadata in split_batches.values()
            if metadata.provider == provider and metadata.model == model
        )
    
    def can_create_batch_for_split(self, split_name: str, provider: str, model: str, 
                                  max_concurrent: int = 5, session_id: Optional[str] = None) -> bool:
        """Check if we can create another batch for this split and model."""
        current_count = self.get_batch_count_for_split(split_name, provider, model, session_id)
        return current_count < max_concurrent
    
    def get_session_recovery_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recovery information for a specific session."""
        if session_id is None:
            current_session = self.session_manager.get_current_session()
            if not current_session:
                return {}
            session_id = current_session.session_id
        
        session_batches = self.get_active_batches_for_session(session_id)
        current_time = time.time()
        
        recovery_info = {
            "session_id": session_id,
            "active_batches": len(session_batches),
            "total_pending_requests": sum(m.request_count for m in session_batches.values()),
            "batches_by_split": {},
            "oldest_batch_age": 0,
            "batch_details": []
        }
        
        # Group by split
        for metadata in session_batches.values():
            split = metadata.split_name
            if split not in recovery_info["batches_by_split"]:
                recovery_info["batches_by_split"][split] = 0
            recovery_info["batches_by_split"][split] += 1
        
        # Batch details
        if session_batches:
            oldest_batch = min(session_batches.values(), key=lambda x: x.created_at)
            recovery_info["oldest_batch_age"] = current_time - oldest_batch.created_at
            
            for metadata in session_batches.values():
                recovery_info["batch_details"].append({
                    "batch_id": metadata.batch_id[:8] + "...",
                    "split": metadata.split_name,
                    "provider": metadata.provider,
                    "model": metadata.model,
                    "status": metadata.status,
                    "age_seconds": current_time - metadata.created_at,
                    "request_count": metadata.request_count,
                    "input_file": metadata.input_file
                })
        
        return recovery_info
    
    def cleanup_session(self, session_id: str, remove_files: bool = False):
        """Clean up all batches for a specific session (safe operation).
        By default, keeps on-disk batch checkpoint files unless remove_files=True.
        """
        if session_id in self.active_batches:
            batch_count = len(self.active_batches[session_id])
            del self.active_batches[session_id]
            print(f"🗑️  Cleaned up {batch_count} active batches for session {session_id}")
        
        if session_id in self.completed_batches:
            completed_count = len(self.completed_batches[session_id])
            del self.completed_batches[session_id]
            print(f"🗑️  Cleaned up {completed_count} completed batches for session {session_id}")
        
        # Optionally remove session batch files
        if remove_files:
            try:
                batch_dir = self._get_session_batch_dir(session_id)
                if batch_dir.exists():
                    import shutil
                    shutil.rmtree(batch_dir)
                    print(f"🗑️  Removed batch directory for session {session_id}")
            except Exception as e:
                print(f"⚠️  Could not remove batch directory for session {session_id}: {e}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics across all sessions."""
        all_active = sum(len(batches) for batches in self.active_batches.values())
        all_completed = sum(len(batches) for batches in self.completed_batches.values())
        
        return {
            "total_active_batches": all_active,
            "total_completed_batches": all_completed,
            "active_sessions": len(self.active_batches),
            "sessions_with_history": len(self.completed_batches),
            "session_breakdown": {
                session_id: {
                    "active": len(batches),
                    "completed": len(self.completed_batches.get(session_id, {}))
                }
                for session_id, batches in self.active_batches.items()
            }
        }
