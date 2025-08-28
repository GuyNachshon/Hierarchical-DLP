"""
Batch Recovery Manager - Handles pause/resume functionality for interrupted operations.

Provides graceful handling of connection losses and interruptions:
- Automatic detection of connection failures
- Safe pause of all active operations
- User prompt for recovery decisions
- Selective or full resume capabilities
- State preservation during interruptions
"""

import asyncio
import json
import time
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .connection_monitor import ConnectionMonitor, ConnectionStatus
from .run_session_manager import RunSessionManager
from .enhanced_batch_tracker import EnhancedBatchTracker
from .split_batch_coordinator import SplitBatchCoordinator


class RecoveryState(Enum):
    """Recovery state enumeration."""
    ACTIVE = "active"
    PAUSED_CONNECTION = "paused_connection"
    PAUSED_USER = "paused_user"
    PAUSED_ERROR = "paused_error"
    RECOVERING = "recovering"
    STOPPED = "stopped"


@dataclass
class RecoveryContext:
    """Context information for recovery operations."""
    pause_reason: str
    pause_timestamp: float
    active_splits: List[str]
    active_batches: Dict[str, Any]
    user_intervention_required: bool = False
    recovery_options: List[str] = None
    
    def __post_init__(self):
        if self.recovery_options is None:
            self.recovery_options = []


class BatchRecoveryManager:
    """Manages pause/resume functionality for batch operations."""
    
    def __init__(self, 
                 session_manager: RunSessionManager,
                 batch_tracker: EnhancedBatchTracker,
                 connection_monitor: ConnectionMonitor,
                 split_coordinator: Optional[SplitBatchCoordinator] = None):
        self.session_manager = session_manager
        self.batch_tracker = batch_tracker
        self.connection_monitor = connection_monitor
        self.split_coordinator = split_coordinator
        
        # Recovery state
        self.current_state = RecoveryState.ACTIVE
        self.recovery_context: Optional[RecoveryContext] = None
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start in unpaused state
        
        # Callbacks
        self.pause_callbacks: List[Callable] = []
        self.resume_callbacks: List[Callable] = []
        
        # Configuration
        self.auto_pause_on_connection_failure = True
        self.max_recovery_attempts = 3
        self.recovery_attempt_delay = 30.0  # seconds
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Setup connection monitoring callbacks
        self._setup_connection_callbacks()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n🚨 Received signal {signum}, initiating graceful pause...")
            asyncio.create_task(self.pause_operations(
                reason=f"Signal {signum} received",
                user_intervention_required=True
            ))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _setup_connection_callbacks(self):
        """Setup connection monitoring callbacks."""
        self.connection_monitor.add_failure_callback(self._on_connection_failure)
        self.connection_monitor.add_recovery_callback(self._on_connection_recovery)
    
    async def _on_connection_failure(self, results: Dict[str, Any]):
        """Called when connection monitoring detects failures."""
        if self.auto_pause_on_connection_failure and self.current_state == RecoveryState.ACTIVE:
            failed_connections = [
                name for name, result in results.items() 
                if result.status == ConnectionStatus.FAILED
            ]
            
            await self.pause_operations(
                reason=f"Connection failure detected: {', '.join(failed_connections)}",
                user_intervention_required=True
            )
    
    async def _on_connection_recovery(self, results: Dict[str, Any]):
        """Called when connections recover."""
        if self.current_state == RecoveryState.PAUSED_CONNECTION:
            print("🟢 Connections recovered. Ready to resume operations.")
            self.recovery_context.recovery_options = [
                "resume_all", "resume_selective", "abort"
            ]
    
    async def pause_operations(self, reason: str, user_intervention_required: bool = False):
        """Pause all active batch operations."""
        if self.current_state != RecoveryState.ACTIVE:
            return  # Already paused
        
        print(f"⏸️  Pausing operations: {reason}")
        
        # Collect current state
        active_splits = []
        active_batches = {}
        
        if self.split_coordinator:
            active_splits = list(self.split_coordinator.get_all_progress().keys())
        
        active_batches = self.batch_tracker.get_all_active_batches()
        
        # Create recovery context
        self.recovery_context = RecoveryContext(
            pause_reason=reason,
            pause_timestamp=time.time(),
            active_splits=active_splits,
            active_batches={k: v.to_dict() for k, v in active_batches.items()},
            user_intervention_required=user_intervention_required,
            recovery_options=["resume", "abort"] if not user_intervention_required else []
        )
        
        # Update state
        if "connection" in reason.lower():
            self.current_state = RecoveryState.PAUSED_CONNECTION
        elif user_intervention_required:
            self.current_state = RecoveryState.PAUSED_USER
        else:
            self.current_state = RecoveryState.PAUSED_ERROR
        
        # Signal pause to all operations
        self.pause_event.clear()
        
        # Save recovery state
        await self._save_recovery_state()
        
        # Notify callbacks
        for callback in self.pause_callbacks:
            try:
                await callback(self.recovery_context)
            except Exception as e:
                print(f"⚠️  Pause callback error: {e}")
        
        print(f"⏸️  Operations paused. Active splits: {len(active_splits)}, Active batches: {len(active_batches)}")
        
        # If user intervention required, wait for user input
        if user_intervention_required:
            await self._wait_for_user_intervention()
    
    async def _wait_for_user_intervention(self):
        """Wait for user intervention and handle their choice."""
        print("🤔 User intervention required. Operations are paused.")
        print("Options:")
        print("  1. Resume all operations")
        print("  2. Resume selective splits")  
        print("  3. Check connection status")
        print("  4. Abort and exit")
        print("  5. Show recovery information")
        
        while self.current_state in [RecoveryState.PAUSED_USER, RecoveryState.PAUSED_CONNECTION]:
            try:
                print("\\nEnter choice (1-5): ", end="", flush=True)
                
                # In a real implementation, you might want to use aioconsole for async input
                # For now, we'll use a simple approach
                choice = await asyncio.to_thread(input)
                
                if choice == "1":
                    await self.resume_operations()
                    break
                elif choice == "2":
                    await self._handle_selective_resume()
                    break
                elif choice == "3":
                    await self._show_connection_status()
                elif choice == "4":
                    await self.abort_operations()
                    break
                elif choice == "5":
                    await self._show_recovery_info()
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\\n🛑 Aborting operations...")
                await self.abort_operations()
                break
            except Exception as e:
                print(f"Error handling input: {e}")
    
    async def _handle_selective_resume(self):
        """Handle selective resume of splits."""
        if not self.recovery_context:
            return
        
        active_splits = self.recovery_context.active_splits
        if not active_splits:
            print("No active splits to resume.")
            await self.resume_operations()
            return
        
        print("\\nActive splits:")
        for i, split in enumerate(active_splits, 1):
            progress = self.split_coordinator.get_split_progress(split) if self.split_coordinator else None
            if progress:
                print(f"  {i}. {split} ({progress.completed_examples}/{progress.target_size} examples)")
            else:
                print(f"  {i}. {split}")
        
        print("Enter split numbers to resume (comma-separated), or 'all': ", end="", flush=True)
        selection = await asyncio.to_thread(input)
        
        if selection.lower() == 'all':
            await self.resume_operations()
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_splits = [active_splits[i] for i in indices if 0 <= i < len(active_splits)]
                await self.resume_operations(selected_splits)
            except ValueError:
                print("Invalid selection. Resuming all operations.")
                await self.resume_operations()
    
    async def _show_connection_status(self):
        """Show current connection status."""
        results = await self.connection_monitor.check_all_connections()
        print("\\n🔌 Connection Status:")
        for name, result in results.items():
            status_emoji = {"healthy": "🟢", "degraded": "🟡", "failed": "🔴"}.get(result.status.value, "⚪")
            print(f"  {status_emoji} {name}: {result.status.value}")
            if result.error_message:
                print(f"    Error: {result.error_message}")
            print(f"    Response time: {result.response_time:.2f}s")
        print()
    
    async def _show_recovery_info(self):
        """Show detailed recovery information."""
        if not self.recovery_context:
            print("No recovery context available.")
            return
        
        print("\\n📊 Recovery Information:")
        print(f"  Pause reason: {self.recovery_context.pause_reason}")
        print(f"  Pause time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.recovery_context.pause_timestamp))}")
        print(f"  Paused duration: {time.time() - self.recovery_context.pause_timestamp:.1f} seconds")
        print(f"  Active splits: {len(self.recovery_context.active_splits)}")
        print(f"  Active batches: {len(self.recovery_context.active_batches)}")
        
        if self.recovery_context.active_splits:
            print("  Splits:")
            for split in self.recovery_context.active_splits:
                print(f"    - {split}")
        
        session_info = self.batch_tracker.get_session_recovery_info()
        print(f"  Pending requests: {session_info.get('total_pending_requests', 0)}")
        print()
    
    async def resume_operations(self, selected_splits: Optional[List[str]] = None):
        """Resume paused operations."""
        if self.current_state == RecoveryState.ACTIVE:
            return  # Already active
        
        print("🔄 Resuming operations...")
        
        # Check connections first
        connection_results = await self.connection_monitor.check_all_connections()
        if not self.connection_monitor.are_required_connections_healthy():
            failed = self.connection_monitor.get_failed_connections()
            print(f"❌ Cannot resume: Required connections still failed: {', '.join(failed)}")
            return False
        
        self.current_state = RecoveryState.RECOVERING
        
        # Resume specific splits if specified
        if selected_splits and self.split_coordinator:
            for split in selected_splits:
                if split not in self.recovery_context.active_splits:
                    continue
                print(f"📍 Resuming {split} split...")
                # The split coordinator will handle resuming from its saved state
        
        # Clear pause event
        self.pause_event.set()
        self.current_state = RecoveryState.ACTIVE
        
        # Clear recovery context
        self.recovery_context = None
        
        # Remove recovery state file
        await self._clear_recovery_state()
        
        # Notify callbacks
        for callback in self.resume_callbacks:
            try:
                await callback()
            except Exception as e:
                print(f"⚠️  Resume callback error: {e}")
        
        print("▶️  Operations resumed successfully")
        return True
    
    async def abort_operations(self):
        """Abort all operations and exit."""
        print("🛑 Aborting all operations...")
        
        self.current_state = RecoveryState.STOPPED
        
        # Cancel all active batches
        if self.recovery_context:
            for batch_id in self.recovery_context.active_batches:
                self.batch_tracker.complete_batch(
                    batch_id, 
                    success=False, 
                    error_message="Operation aborted by user"
                )
        
        # Update session status
        self.session_manager.update_session_status("interrupted", "User aborted operations")
        
        # Clear recovery state
        await self._clear_recovery_state()
        
        print("🛑 Operations aborted. Exiting...")
        sys.exit(1)
    
    async def wait_if_paused(self):
        """Wait if operations are currently paused."""
        await self.pause_event.wait()
    
    def is_paused(self) -> bool:
        """Check if operations are currently paused."""
        return self.current_state != RecoveryState.ACTIVE
    
    def get_recovery_state(self) -> RecoveryState:
        """Get current recovery state."""
        return self.current_state
    
    def get_recovery_context(self) -> Optional[RecoveryContext]:
        """Get current recovery context."""
        return self.recovery_context
    
    async def _save_recovery_state(self):
        """Save recovery state to file."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir or not self.recovery_context:
            return
        
        recovery_file = session_dir / "recovery_state.json"
        
        recovery_data = {
            "state": self.current_state.value,
            "context": {
                "pause_reason": self.recovery_context.pause_reason,
                "pause_timestamp": self.recovery_context.pause_timestamp,
                "active_splits": self.recovery_context.active_splits,
                "active_batches": self.recovery_context.active_batches,
                "user_intervention_required": self.recovery_context.user_intervention_required,
                "recovery_options": self.recovery_context.recovery_options
            }
        }
        
        try:
            with open(recovery_file, 'w') as f:
                json.dump(recovery_data, f, indent=2)
            print(f"💾 Recovery state saved: {recovery_file}")
        except Exception as e:
            print(f"⚠️  Failed to save recovery state: {e}")
    
    async def _clear_recovery_state(self):
        """Clear saved recovery state."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            return
        
        recovery_file = session_dir / "recovery_state.json"
        if recovery_file.exists():
            recovery_file.unlink()
    
    async def load_recovery_state(self) -> bool:
        """Load and restore from saved recovery state."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            return False
        
        recovery_file = session_dir / "recovery_state.json"
        if not recovery_file.exists():
            return False
        
        try:
            with open(recovery_file, 'r') as f:
                recovery_data = json.load(f)
            
            # Restore state
            self.current_state = RecoveryState(recovery_data["state"])
            context_data = recovery_data["context"]
            
            self.recovery_context = RecoveryContext(
                pause_reason=context_data["pause_reason"],
                pause_timestamp=context_data["pause_timestamp"],
                active_splits=context_data["active_splits"],
                active_batches=context_data["active_batches"],
                user_intervention_required=context_data["user_intervention_required"],
                recovery_options=context_data["recovery_options"]
            )
            
            if self.current_state != RecoveryState.ACTIVE:
                self.pause_event.clear()
            
            pause_duration = time.time() - self.recovery_context.pause_timestamp
            print(f"📂 Loaded recovery state: {self.current_state.value}")
            print(f"⏰ Operations were paused for {pause_duration:.1f} seconds")
            print(f"🔄 Reason: {self.recovery_context.pause_reason}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load recovery state: {e}")
            return False
    
    def add_pause_callback(self, callback: Callable):
        """Add callback to be called when operations are paused."""
        self.pause_callbacks.append(callback)
    
    def add_resume_callback(self, callback: Callable):
        """Add callback to be called when operations are resumed."""
        self.resume_callbacks.append(callback)