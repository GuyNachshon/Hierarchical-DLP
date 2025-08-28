"""
Enhanced coordinator with run session management and robust batch processing.

Integrates the new batch management system with:
- Run session isolation
- Split-aware processing
- Connection monitoring
- Recovery/pause functionality
"""

import os
import time
import asyncio
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from .config import AgenticConfig
from .agents import (
    ManagerAgent, LegalAgent, FinanceAgent, HRAgent, SecurityAgent,
    CasualAgent, CleanBusinessAgent, ObfuscationSpecialist, ConversationalAgent
)
from .batch.batch_processor import BatchProcessor
from .batch.run_session_manager import RunSessionManager
from .batch.enhanced_batch_tracker import EnhancedBatchTracker
from .batch.connection_monitor import ConnectionMonitor
from .batch.split_batch_coordinator import SplitBatchCoordinator, SplitGenerationPlan
from .batch.batch_recovery_manager import BatchRecoveryManager
from .data.validators import QualityValidator
from .data.converters import DLPFormatConverter


class EnhancedAgenticDataGenerator:
    """Enhanced coordinator with run session management and robust batch processing."""
    
    def __init__(self, config: AgenticConfig):
        self.config = config
        
        # Initialize batch management system
        self.session_manager = RunSessionManager()
        self.connection_monitor = ConnectionMonitor()
        self.batch_tracker = EnhancedBatchTracker(self.session_manager)
        self.batch_processor = BatchProcessor(config)
        self.split_coordinator = SplitBatchCoordinator(
            self.session_manager, 
            self.batch_tracker, 
            self.batch_processor
        )
        self.recovery_manager = BatchRecoveryManager(
            self.session_manager,
            self.batch_tracker,
            self.connection_monitor,
            self.split_coordinator
        )
        
        # Data processing
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
            "agent_stats": {name: {"requested": 0, "generated": 0} for name in self.agents.keys()},
            "split_stats": {"train": 0, "val": 0, "test": 0},
            "batch_stats": {"total_batches": 0, "successful_batches": 0, "failed_batches": 0}
        }
        
        # Session tracking
        self.current_session = None
        self.monitoring_task = None
    
    async def generate_dataset(self) -> None:
        """Generate complete dataset using enhanced split-aware approach."""
        try:
            # Create run session
            self.current_session = self.session_manager.create_run_session(self.config)
            print(f"🚀 Starting Enhanced Agentic Data Generator v2.1")
            print(f"📋 Session ID: {self.current_session.session_id}")
            print(f"📊 Target: {self._get_total_target()} examples")
            
            # Check for existing recovery state
            if await self.recovery_manager.load_recovery_state():
                print("🔄 Found existing recovery state. Continuing from where we left off...")
                if self.recovery_manager.is_paused():
                    print("⏸️  Operations are currently paused. Waiting for user intervention...")
                    return
            
            # Start connection monitoring
            self.monitoring_task = asyncio.create_task(self.connection_monitor.start_monitoring())
            
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Generate split plans
            split_plans = await self._create_split_generation_plans()
            
            # Generate all splits using the split coordinator
            split_results = {}
            
            print("🎯 Starting split-aware generation...")
            for plan in split_plans:
                # Check if we need to pause
                await self.recovery_manager.wait_if_paused()
                
                print(f"📊 Generating {plan.split_name} split...")
                try:
                    split_examples = await self.split_coordinator.generate_split_dataset(plan)
                    split_results[plan.split_name] = split_examples
                    self.global_stats["split_stats"][plan.split_name] = len(split_examples)
                    
                    print(f"✅ {plan.split_name} split completed: {len(split_examples)} examples")
                    
                except Exception as e:
                    print(f"❌ {plan.split_name} split failed: {e}")
                    # The recovery manager will handle this appropriately
                    split_results[plan.split_name] = []
            
            # Create final dataset files
            await self.split_coordinator.create_final_split_files(split_results, self.config.output_dir)
            
            # Save comprehensive statistics
            await self._save_enhanced_statistics(split_results)
            
            # Update session status
            self.session_manager.update_session_status("completed")
            
            print("🎉 Enhanced dataset generation completed successfully!")
            print(f"📁 Dataset saved to: {self.config.output_dir}")
            print(f"📊 Session: {self.current_session.session_id}")
            
        except Exception as e:
            print(f"❌ Dataset generation failed: {e}")
            self.session_manager.update_session_status("failed", str(e))
            raise
        
        finally:
            # Stop monitoring
            if self.monitoring_task:
                self.connection_monitor.stop_monitoring()
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
    
    async def _create_split_generation_plans(self) -> List[SplitGenerationPlan]:
        """Create generation plans for each split."""
        plans = []
        
        # Create requests for each split
        splits = [
            ("train", self.config.train_size),
            ("val", self.config.val_size), 
            ("test", self.config.test_size)
        ]
        
        for split_name, split_size in splits:
            print(f"📝 Creating generation plan for {split_name} split ({split_size} examples)...")
            
            # Create generation requests using the manager agent
            requests = await self.manager.create_generation_plan(split_name, split_size)
            
            plan = SplitGenerationPlan(
                split_name=split_name,
                target_size=split_size,
                requests=requests,
                batch_size=getattr(self.config, 'batch_size', 50)
            )
            
            plans.append(plan)
            print(f"📋 {split_name}: {len(requests)} requests, ~{plan.estimated_batches} batches")
        
        return plans
    
    async def _save_enhanced_statistics(self, split_results: Dict[str, List[Dict]]):
        """Save comprehensive statistics about the generation process."""
        session_dir = self.session_manager.get_session_directory()
        if not session_dir:
            return
        
        stats_file = session_dir / "generation_stats.json"
        
        # Gather comprehensive statistics
        connection_stats = self.connection_monitor.get_monitoring_stats()
        batch_stats = self.batch_tracker.get_global_stats()
        session_stats = self.session_manager.get_session_stats()
        
        comprehensive_stats = {
            "session": {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time,
                "end_time": time.time(),
                "duration_seconds": time.time() - self.current_session.start_time,
                "config": self.current_session.config
            },
            "generation": {
                "total_examples": sum(len(examples) for examples in split_results.values()),
                "split_breakdown": {
                    split: len(examples) for split, examples in split_results.items()
                },
                "quality_stats": self.global_stats,
                "agent_distribution": self._calculate_agent_distribution(split_results)
            },
            "batch_processing": {
                "total_batches_created": batch_stats.get("total_active_batches", 0) + batch_stats.get("total_completed_batches", 0),
                "successful_batches": batch_stats.get("total_completed_batches", 0),
                "session_breakdown": batch_stats.get("session_breakdown", {}),
                "batch_efficiency": self._calculate_batch_efficiency()
            },
            "connection_monitoring": {
                "monitoring_duration": connection_stats.get("last_check", 0) - self.current_session.start_time if connection_stats.get("last_check") else 0,
                "connection_health": connection_stats.get("required_connections_healthy", False),
                "connection_details": connection_stats.get("connection_details", {})
            },
            "system_info": {
                "recovery_used": self.recovery_manager.get_recovery_context() is not None,
                "session_stats": session_stats
            }
        }
        
        # Save to session directory
        with open(stats_file, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        
        # Also save to output directory for easy access
        output_stats_file = Path(self.config.output_dir) / "generation_stats.json"
        with open(output_stats_file, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        
        print(f"📊 Comprehensive statistics saved:")
        print(f"   Session: {stats_file}")
        print(f"   Output: {output_stats_file}")
    
    def _calculate_agent_distribution(self, split_results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Calculate distribution of examples by agent type."""
        agent_counts = {}
        
        for split_examples in split_results.values():
            for example in split_examples:
                agent_type = example.get("_metadata", {}).get("agent_type", "unknown")
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        return agent_counts
    
    def _calculate_batch_efficiency(self) -> Dict[str, float]:
        """Calculate batch processing efficiency metrics."""
        batch_stats = self.batch_tracker.get_global_stats()
        
        total_batches = batch_stats.get("total_active_batches", 0) + batch_stats.get("total_completed_batches", 0)
        successful_batches = batch_stats.get("total_completed_batches", 0)
        
        if total_batches == 0:
            return {"success_rate": 0.0, "failure_rate": 0.0}
        
        success_rate = successful_batches / total_batches
        failure_rate = 1.0 - success_rate
        
        return {
            "success_rate": round(success_rate, 3),
            "failure_rate": round(failure_rate, 3),
            "total_batches": total_batches
        }
    
    def _get_total_target(self) -> int:
        """Get total target size for dataset."""
        return self.config.train_size + self.config.val_size + self.config.test_size
    
    async def cleanup_session(self, cleanup_files: bool = False):
        """Clean up the current session."""
        if not self.current_session:
            return
        
        print(f"🧹 Cleaning up session: {self.current_session.session_id}")
        
        # Clean up batch tracking
        self.batch_tracker.cleanup_session(self.current_session.session_id)
        
        # Stop monitoring
        if self.monitoring_task:
            self.connection_monitor.stop_monitoring()
            self.monitoring_task.cancel()
        
        # Clean up files if requested
        if cleanup_files:
            session_dir = self.session_manager.get_session_directory()
            if session_dir and session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                print(f"🗑️  Removed session directory: {session_dir}")
        
        print("✅ Session cleanup completed")
    
    async def resume_from_session(self, session_id: str) -> bool:
        """Resume generation from an existing session."""
        session = self.session_manager.load_run_session(session_id)
        if not session:
            print(f"❌ Session {session_id} not found")
            return False
        
        self.current_session = session
        
        # Load recovery state if exists
        if await self.recovery_manager.load_recovery_state():
            print("🔄 Recovery state loaded")
            
            # Start monitoring
            self.monitoring_task = asyncio.create_task(self.connection_monitor.start_monitoring())
            
            # Continue generation process
            await self.generate_dataset()
            return True
        else:
            print("ℹ️  No recovery state found, starting fresh generation")
            await self.generate_dataset()
            return True
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current session."""
        if not self.current_session:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "status": self.current_session.status,
            "start_time": self.current_session.start_time,
            "config": self.current_session.config,
            "recovery_state": self.recovery_manager.get_recovery_state().value if self.recovery_manager else None,
            "connection_status": self.connection_monitor.get_connection_status(),
            "batch_stats": self.batch_tracker.get_session_recovery_info(),
            "split_progress": self.split_coordinator.get_all_progress()
        }
    
    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = self.session_manager.list_all_sessions()
        return [
            {
                "session_id": session.session_id,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": (session.end_time or time.time()) - session.start_time,
                "config": session.config
            }
            for session in sessions
        ]