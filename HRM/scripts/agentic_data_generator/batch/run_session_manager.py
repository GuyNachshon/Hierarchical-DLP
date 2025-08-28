"""
Run Session Manager - Provides run isolation for batch operations.

Each data generation run gets a unique session ID to prevent cross-contamination
of batches from different runs and enable safe cleanup operations.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RunSession:
    """Information about a data generation run session."""
    session_id: str
    start_time: float
    config: Dict[str, Any]
    output_dir: str
    splits: List[str]
    status: str = "active"  # active, completed, failed, interrupted
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunSession':
        """Create from dictionary."""
        return cls(**data)


class RunSessionManager:
    """Manages run sessions for isolated batch processing."""
    
    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = Path(base_data_dir)
        self.runs_dir = self.base_data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_session: Optional[RunSession] = None
        self.session_dir: Optional[Path] = None
    
    def create_run_session(self, config: Any) -> RunSession:
        """Create a new run session."""
        # Generate unique session ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_uuid = str(uuid.uuid4())[:8]
        session_id = f"run_{timestamp}_{session_uuid}"
        
        # Create session object
        session = RunSession(
            session_id=session_id,
            start_time=time.time(),
            config={
                "train_size": getattr(config, 'train_size', 0),
                "val_size": getattr(config, 'val_size', 0),
                "test_size": getattr(config, 'test_size', 0),
                "output_dir": getattr(config, 'output_dir', 'unknown'),
                "enable_batch_api": getattr(config, 'enable_batch_api', False),
                "max_concurrent_agents": getattr(config, 'max_concurrent_agents', 10),
            },
            output_dir=getattr(config, 'output_dir', 'unknown'),
            splits=["train", "val", "test"]
        )
        
        # Create session directory
        self.session_dir = self.runs_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.session_dir / "checkpoints").mkdir(exist_ok=True)
        (self.session_dir / "batch_inputs").mkdir(exist_ok=True)
        (self.session_dir / "batch_outputs").mkdir(exist_ok=True)
        
        # Save session info
        session_file = self.session_dir / "session.json"
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        
        self.current_session = session
        
        print(f"🏁 Created run session: {session_id}")
        print(f"📁 Session directory: {self.session_dir}")
        
        return session
    
    def load_run_session(self, session_id: str) -> Optional[RunSession]:
        """Load an existing run session."""
        session_dir = self.runs_dir / session_id
        session_file = session_dir / "session.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            session = RunSession.from_dict(data)
            self.current_session = session
            self.session_dir = session_dir
            
            print(f"📂 Loaded run session: {session_id}")
            return session
            
        except Exception as e:
            print(f"❌ Failed to load session {session_id}: {e}")
            return None
    
    def get_current_session(self) -> Optional[RunSession]:
        """Get the current active session."""
        return self.current_session
    
    def get_session_directory(self) -> Optional[Path]:
        """Get the current session directory."""
        return self.session_dir
    
    def update_session_status(self, status: str, error_message: Optional[str] = None):
        """Update the current session status."""
        if not self.current_session:
            return
        
        self.current_session.status = status
        if error_message:
            self.current_session.error_message = error_message
        
        if status in ["completed", "failed", "interrupted"]:
            self.current_session.end_time = time.time()
        
        # Save updated session
        if self.session_dir:
            session_file = self.session_dir / "session.json"
            with open(session_file, 'w') as f:
                json.dump(self.current_session.to_dict(), f, indent=2)
    
    def list_all_sessions(self) -> List[RunSession]:
        """List all run sessions."""
        sessions = []
        
        for session_dir in self.runs_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("run_"):
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file, 'r') as f:
                            data = json.load(f)
                        sessions.append(RunSession.from_dict(data))
                    except Exception:
                        continue
        
        # Sort by start time, newest first
        sessions.sort(key=lambda x: x.start_time, reverse=True)
        return sessions
    
    def get_active_sessions(self) -> List[RunSession]:
        """Get all sessions that are currently active."""
        return [s for s in self.list_all_sessions() if s.status == "active"]
    
    def cleanup_old_sessions(self, keep_days: int = 30):
        """Clean up old session directories."""
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for session in self.list_all_sessions():
            if session.start_time < cutoff_time and session.status != "active":
                session_dir = self.runs_dir / session.session_id
                if session_dir.exists():
                    import shutil
                    shutil.rmtree(session_dir)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🗑️  Cleaned up {cleaned_count} old session directories")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        sessions = self.list_all_sessions()
        
        stats = {
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s.status == "active"]),
            "completed_sessions": len([s for s in sessions if s.status == "completed"]),
            "failed_sessions": len([s for s in sessions if s.status == "failed"]),
            "interrupted_sessions": len([s for s in sessions if s.status == "interrupted"])
        }
        
        if sessions:
            stats["oldest_session"] = min(s.start_time for s in sessions)
            stats["newest_session"] = max(s.start_time for s in sessions)
        
        return stats
    
    def save_split_input(self, split_name: str, batch_input_data: List[Dict]) -> Path:
        """Save batch input data for a specific split."""
        if not self.session_dir:
            raise RuntimeError("No active session")
        
        input_file = self.session_dir / "batch_inputs" / f"{split_name}_batch_input.jsonl"
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for item in batch_input_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"💾 Saved {len(batch_input_data)} batch inputs for {split_name}: {input_file}")
        return input_file
    
    def load_split_input(self, split_name: str) -> Optional[List[Dict]]:
        """Load batch input data for a specific split."""
        if not self.session_dir:
            return None
        
        input_file = self.session_dir / "batch_inputs" / f"{split_name}_batch_input.jsonl"
        
        if not input_file.exists():
            return None
        
        batch_inputs = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        batch_inputs.append(json.loads(line))
            return batch_inputs
        except Exception as e:
            print(f"❌ Failed to load batch inputs for {split_name}: {e}")
            return None