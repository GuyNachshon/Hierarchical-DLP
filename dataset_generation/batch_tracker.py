#!/usr/bin/env python3
"""
Persistent Batch Tracker for reliable batch API state management.

Handles:
- Persistent storage of active batch IDs
- Recovery of interrupted batches
- Batch timeout and cleanup
- Result mapping and validation
"""

import json
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


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
class BatchMetadata:
    """Metadata for tracking a batch"""
    batch_id: str
    provider: str  # "openai" or "anthropic"
    model: str
    request_count: int
    created_at: float
    timeout_at: float
    status: str = BatchStatus.CREATED.value
    request_indices: List[int] = None  # Original indices for result mapping
    results_retrieved: bool = False
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.request_indices is None:
            self.request_indices = []


class BatchTracker:
    """Persistent batch tracker for reliable state management"""

    def __init__(self, checkpoint_dir: str, batch_timeout: int = 86400):  # 24 hours in seconds
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.batch_timeout = batch_timeout

        # State files
        self.active_batches_file = self.checkpoint_dir / "active_batches.json"
        self.completed_batches_file = self.checkpoint_dir / "completed_batches.json"

        # In-memory state
        self.active_batches: Dict[str, BatchMetadata] = {}
        self.completed_batches: Dict[str, BatchMetadata] = {}

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load existing batch state from disk"""
        # Load active batches
        if self.active_batches_file.exists():
            try:
                with open(self.active_batches_file, 'r') as f:
                    data = json.load(f)
                    self.active_batches = {
                        batch_id: BatchMetadata(**batch_data)
                        for batch_id, batch_data in data.items()
                    }
                # Silently track state - dashboard will show this info
                if not hasattr(self, '_logged_active_count'):
                    self._logged_active_count = len(self.active_batches)
            except Exception as e:
                print(f"⚠️  Failed to load active batches: {e}")
                self.active_batches = {}

        # Load completed batches
        if self.completed_batches_file.exists():
            try:
                with open(self.completed_batches_file, 'r') as f:
                    data = json.load(f)
                    self.completed_batches = {
                        batch_id: BatchMetadata(**batch_data)
                        for batch_id, batch_data in data.items()
                    }
                # Silently track state - dashboard will show this info
                if not hasattr(self, '_logged_completed_count'):
                    self._logged_completed_count = len(self.completed_batches)
            except Exception as e:
                print(f"⚠️  Failed to load completed batches: {e}")
                self.completed_batches = {}

    def _save_state(self):
        """Save current state to disk"""
        try:
            # Save active batches
            with open(self.active_batches_file, 'w') as f:
                json.dump(
                    {batch_id: asdict(metadata) for batch_id, metadata in self.active_batches.items()},
                    f, indent=2
                )

            # Save completed batches (keep last 100 for history)
            recent_completed = dict(list(self.completed_batches.items())[-100:])
            with open(self.completed_batches_file, 'w') as f:
                json.dump(
                    {batch_id: asdict(metadata) for batch_id, metadata in recent_completed.items()},
                    f, indent=2
                )
        except Exception as e:
            print(f"❌ Failed to save batch state: {e}")

    def create_batch(self, batch_id: str, provider: str, model: str,
                     request_count: int, request_indices: List[int]) -> BatchMetadata:
        """Register a new batch for tracking"""
        current_time = time.time()

        metadata = BatchMetadata(
            batch_id=batch_id,
            provider=provider,
            model=model,
            request_count=request_count,
            created_at=current_time,
            timeout_at=current_time + self.batch_timeout,
            request_indices=request_indices.copy()
        )

        self.active_batches[batch_id] = metadata
        self._save_state()

        print(f"📝 Registered batch {batch_id[:8]}... ({provider}/{model}, {request_count} requests)")
        return metadata

    def update_batch_status(self, batch_id: str, status: BatchStatus, error_message: Optional[str] = None):
        """Update batch status"""
        if batch_id in self.active_batches:
            self.active_batches[batch_id].status = status.value
            if error_message:
                self.active_batches[batch_id].error_message = error_message
            self._save_state()

    def complete_batch(self, batch_id: str, success: bool = True, error_message: Optional[str] = None):
        """Mark batch as completed and move to completed list"""
        if batch_id not in self.active_batches:
            return

        metadata = self.active_batches[batch_id]
        metadata.status = BatchStatus.COMPLETED.value if success else BatchStatus.FAILED.value
        metadata.results_retrieved = success
        if error_message:
            metadata.error_message = error_message

        # Move to completed
        self.completed_batches[batch_id] = metadata
        del self.active_batches[batch_id]

        self._save_state()

        status_emoji = "✅" if success else "❌"
        print(f"{status_emoji} Batch {batch_id[:8]}... completed ({'success' if success else 'failed'})")

    def get_active_batches(self) -> Dict[str, BatchMetadata]:
        """Get all active batches"""
        return self.active_batches.copy()

    def get_timed_out_batches(self) -> List[BatchMetadata]:
        """Get batches that have exceeded timeout"""
        current_time = time.time()
        return [
            metadata for metadata in self.active_batches.values()
            if current_time > metadata.timeout_at and metadata.status not in [
                BatchStatus.COMPLETED.value, BatchStatus.FAILED.value
            ]
        ]

    def cleanup_timed_out_batches(self) -> List[str]:
        """Mark timed-out batches as expired and return their IDs"""
        timed_out = self.get_timed_out_batches()
        expired_ids = []

        for metadata in timed_out:
            print(f"⏰ Batch {metadata.batch_id[:8]}... timed out after {self.batch_timeout}s")
            self.complete_batch(metadata.batch_id, success=False, error_message="Batch timed out")
            expired_ids.append(metadata.batch_id)

        return expired_ids

    def get_active_batch_count(self) -> int:
        """Get count of active batches"""
        return len(self.active_batches)

    def get_model_batch_count(self, provider: str, model: str) -> int:
        """Get count of active batches for a specific model"""
        return sum(
            1 for metadata in self.active_batches.values()
            if metadata.provider == provider and metadata.model == model
        )

    def can_create_batch(self, provider: str, model: str, max_concurrent: int = 5) -> bool:
        """Check if we can create another batch for this model"""
        current_count = self.get_model_batch_count(provider, model)
        return current_count < max_concurrent

    def get_batch_results_mapping(self, batch_id: str) -> Optional[List[int]]:
        """Get the original request indices for result mapping"""
        if batch_id in self.active_batches:
            return self.active_batches[batch_id].request_indices
        elif batch_id in self.completed_batches:
            return self.completed_batches[batch_id].request_indices
        return None

    def get_recovery_info(self) -> Dict:
        """Get information about batches that need recovery"""
        current_time = time.time()
        recovery_info = {
            "active_batches": len(self.active_batches),
            "timed_out_batches": len(self.get_timed_out_batches()),
            "total_pending_requests": sum(m.request_count for m in self.active_batches.values()),
            "oldest_batch_age": 0,
            "batch_details": []
        }

        if self.active_batches:
            oldest_batch = min(self.active_batches.values(), key=lambda x: x.created_at)
            recovery_info["oldest_batch_age"] = current_time - oldest_batch.created_at

            for metadata in self.active_batches.values():
                recovery_info["batch_details"].append({
                    "batch_id": metadata.batch_id[:8] + "...",
                    "provider": metadata.provider,
                    "model": metadata.model,
                    "status": metadata.status,
                    "age_seconds": current_time - metadata.created_at,
                    "request_count": metadata.request_count
                })

        return recovery_info

    def clear_completed_history(self):
        """Clear completed batch history (for cleanup)"""
        self.completed_batches.clear()
        if self.completed_batches_file.exists():
            self.completed_batches_file.unlink()
        print("🗑️  Cleared completed batch history")
