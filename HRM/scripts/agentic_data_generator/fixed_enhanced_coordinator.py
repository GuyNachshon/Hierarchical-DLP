"""
Fixed Enhanced coordinator - One batch per split with consistent model usage.

Key fixes applied:
- Each split becomes exactly one batch (unless >140K entries)  
- Each batch uses exactly one model for all requests
- Maintains all existing recovery and session management features
"""

import os
import time
import asyncio
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from config import AgenticConfig
from agents import (
    ManagerAgent, LegalAgent, FinanceAgent, HRAgent, SecurityAgent,
    CasualAgent, CleanBusinessAgent, ObfuscationSpecialist, ConversationalAgent, AugmentationAgent
)
from batch.fixed_batch_processor import FixedBatchProcessor
from batch.run_session_manager import RunSessionManager
from batch.enhanced_batch_tracker import EnhancedBatchTracker
from batch.connection_monitor import ConnectionMonitor
from batch.fixed_split_batch_coordinator import FixedSplitBatchCoordinator, SplitGenerationPlan
from batch.batch_recovery_manager import BatchRecoveryManager
# Import our custom DLP converter (skip missing validators for now)
try:
    from data.dlp_converter import DLPFormatConverter
except ImportError:
    # Fallback - will create minimal converter if needed
    DLPFormatConverter = None

# Import augmentation system
try:
    from augmentation import DatasetAugmentor, AugmentationConfig, RiskScenarioGenerator
except ImportError:
    DatasetAugmentor = None
    AugmentationConfig = None
    RiskScenarioGenerator = None


class FixedEnhancedAgenticDataGenerator:
    """Fixed enhanced coordinator with proper batch and model management."""
    
    def __init__(self, config: AgenticConfig):
        self.config = config
        
        # Update config to reflect fixed batch behavior
        self.config.max_batch_size = getattr(config, 'max_batch_size', 140000)  # 140K max per batch
        
        # Initialize batch management system with fixed components
        self.session_manager = RunSessionManager()
        self.connection_monitor = ConnectionMonitor()
        self.batch_tracker = EnhancedBatchTracker(self.session_manager)
        self.batch_processor = FixedBatchProcessor(config)  # Fixed processor
        self.split_coordinator = FixedSplitBatchCoordinator(  # Fixed coordinator
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
        
        # Expose pause event to components that may poll/wait
        try:
            self.config.recovery_pause_event = self.recovery_manager.pause_event
        except Exception:
            pass
        
        # Data processing
        self.dlp_converter = DLPFormatConverter() if DLPFormatConverter else None
        
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
        self.augmentation_agent = AugmentationAgent(config)
        
        # Enhanced statistics
        self.global_stats = {
            "total_requested": 0,
            "total_generated": 0,
            "quality_rejections": 0,
            "agent_stats": {name: {"requested": 0, "generated": 0} for name in self.agents.keys()},
            "split_stats": {"train": 0, "val": 0, "test": 0},
            "batch_stats": {"total_batches": 0, "successful_batches": 0, "failed_batches": 0},
            "model_stats": {}  # Track which models were used
        }
        
        # Session tracking
        self.current_session = None
        self.monitoring_task = None
    
    async def generate_dataset(self) -> None:
        """Generate complete dataset using fixed split-aware approach."""
        try:
            # Create run session
            self.current_session = self.session_manager.create_run_session(self.config)
            print(f"🚀 Starting Fixed Enhanced Agentic Data Generator v2.2")
            print(f"📋 Session ID: {self.current_session.session_id}")
            print(f"📊 Target: {self._get_total_target()} examples")
            print(f"🔧 Fixed batch mode: One batch per split (max {self.config.max_batch_size:,} entries)")
            
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
            
            # Generate split plans (each split = one batch)
            split_plans = await self._create_fixed_split_generation_plans()
            
            # Show batch planning
            self._show_batch_planning_info(split_plans)
            
            # Generate all splits using the fixed coordinator
            print("🎯 Starting fixed split-aware generation (one batch per split)...")
            split_results = await self.split_coordinator.generate_all_splits(split_plans)
            
            # Optional augmentation pass (on-the-fly) before finalization
            if getattr(self.config, 'enable_augmentation', False):
                print("✨ Running augmentation pass on generated examples...")
                split_results = await self._augment_split_results(split_results)

            # Update statistics
            for split_name, examples in split_results.items():
                self.global_stats["split_stats"][split_name] = len(examples)
                self._update_model_stats(examples)
            
            # If auto retrieval is disabled, do not finalize files; print operator guidance
            if getattr(self.config, 'auto_retrieve_batches', True):
                # Create final dataset files
                await self.split_coordinator.create_final_split_files(split_results, self.config.output_dir)
            else:
                print("⏭️  Auto-retrieval disabled: submitted batches only. Skipping finalization.")
                print("ℹ️  Use your batch retrieval tooling to fetch results and rebuild split files.")
            
            # Save comprehensive statistics
            await self._save_fixed_enhanced_statistics(split_results)
            
            # Update session status
            self.session_manager.update_session_status("completed")
            
            print("🎉 Fixed enhanced dataset generation completed successfully!")
            print(f"📁 Dataset saved to: {self.config.output_dir}")
            print(f"📊 Session: {self.current_session.session_id}")
            self._show_completion_summary(split_results)
            
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

    async def _augment_split_results(self, split_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Augment existing split results using comprehensive augmentation system."""
        if not DatasetAugmentor:
            print("⚠️  Augmentation module not available, skipping augmentation")
            return split_results
        
        # Create augmentation configuration
        augmentation_config = AugmentationConfig(
            augmentation_ratio=getattr(self.config, 'augmentation_ratio', 2.0),
            recipient_variation_prob=0.7,
            content_variation_prob=0.5,
            thread_variation_prob=0.4,
            obfuscation_prob=0.3
        )
        
        # Initialize augmentor
        augmentor = DatasetAugmentor(augmentation_config)
        risk_generator = RiskScenarioGenerator(augmentor)
        
        print(f"✨ Starting comprehensive augmentation (ratio: {augmentation_config.augmentation_ratio}x)")
        
        augmented_results = {}
        
        for split_name, examples in split_results.items():
            if not examples:
                augmented_results[split_name] = []
                continue
            
            print(f"   Processing {split_name} split: {len(examples)} base examples")
            
            # Apply comprehensive augmentation
            augmented_examples = augmentor.augment_batch(examples)
            print(f"      Generated {len(augmented_examples)} total examples (including originals)")
            
            # Generate specific risk scenarios  
            # Check both 'meta' and '_metadata' fields for compatibility
            authorized_examples = [ex for ex in examples if 
                                 ex.get('meta', {}).get('risk_type') == 'authorized' or
                                 ex.get('_metadata', {}).get('agent_type') in ['finance', 'hr', 'legal']]
            if authorized_examples:
                violation_scenarios = risk_generator.create_violation_scenarios(authorized_examples[:10])  # Limit to avoid explosion
                augmented_examples.extend(violation_scenarios)
                print(f"      Added {len(violation_scenarios)} violation scenarios")
            
            # Generate obfuscated scenarios
            if len(examples) > 5:  # Only if we have enough base examples
                obfuscated_scenarios = risk_generator.create_obfuscated_scenarios(examples[:5])
                augmented_examples.extend(obfuscated_scenarios)
                print(f"      Added {len(obfuscated_scenarios)} obfuscated scenarios")
            
            augmented_results[split_name] = augmented_examples
            print(f"   ✅ {split_name}: {len(examples)} → {len(augmented_examples)} examples")
        
        total_original = sum(len(examples) for examples in split_results.values())
        total_augmented = sum(len(examples) for examples in augmented_results.values())
        print(f"🎯 Augmentation complete: {total_original} → {total_augmented} examples ({total_augmented/total_original:.1f}x expansion)")
        
        return augmented_results
    
    async def _create_fixed_split_generation_plans(self) -> List[SplitGenerationPlan]:
        """Create generation plans for each split (each split = one batch)."""
        plans = []
        
        # Create requests for each split
        splits = [
            ("train", self.config.train_size),
            ("val", self.config.val_size), 
            ("test", self.config.test_size)
        ]
        
        for split_name, split_size in splits:
            print(f"📝 Creating fixed generation plan for {split_name} split ({split_size:,} examples)...")
            
            # Create generation requests using the manager agent
            requests = await self.manager.create_generation_plan(split_name, split_size)
            
            plan = SplitGenerationPlan(
                split_name=split_name,
                target_size=split_size,
                requests=requests,
                max_batch_size=self.config.max_batch_size
            )
            
            plans.append(plan)
            
            # Show planning info
            batch_info = "single batch" if len(requests) <= self.config.max_batch_size else f"{plan.estimated_batches} batches"
            print(f"📋 {split_name}: {len(requests):,} requests → {batch_info}")
        
        return plans
    
    def _show_batch_planning_info(self, split_plans: List[SplitGenerationPlan]):
        """Show detailed batch planning information."""
        print("\n🔍 Fixed Batch Planning Summary:")
        print("=" * 50)
        
        total_requests = sum(len(plan.requests) for plan in split_plans)
        total_batches = sum(plan.estimated_batches for plan in split_plans)
        
        for plan in split_plans:
            batch_info = "✅ Single batch" if plan.estimated_batches == 1 else f"⚠️  {plan.estimated_batches} batches (large split)"
            print(f"  {plan.split_name:>5}: {len(plan.requests):>6,} requests → {batch_info}")
        
        print("-" * 50)
        print(f"  Total: {total_requests:>6,} requests → {total_batches} batches")
        print(f"  Each batch uses ONE consistent model")
        print(f"  Max batch size: {self.config.max_batch_size:,} entries")
        print()
    
    def _show_completion_summary(self, split_results: Dict[str, List[Dict]]):
        """Show completion summary with model usage."""
        print("\n🎯 Generation Summary:")
        print("=" * 40)
        
        for split_name, examples in split_results.items():
            if examples:
                model_info = examples[0].get("_metadata", {})
                provider = model_info.get("provider", "unknown")
                model = model_info.get("model", "unknown")
                print(f"  {split_name:>5}: {len(examples):>5,} examples using {provider}/{model}")
            else:
                print(f"  {split_name:>5}: {0:>5,} examples (failed)")
        
        total_examples = sum(len(examples) for examples in split_results.values())
        print(f"  Total: {total_examples:>5,} examples")
        print()
    
    def _update_model_stats(self, examples: List[Dict]):
        """Update model usage statistics."""
        for example in examples:
            metadata = example.get("_metadata", {})
            provider = metadata.get("provider", "unknown")
            model = metadata.get("model", "unknown")
            
            model_key = f"{provider}/{model}"
            if model_key not in self.global_stats["model_stats"]:
                self.global_stats["model_stats"][model_key] = 0
            self.global_stats["model_stats"][model_key] += 1
    
    async def _save_fixed_enhanced_statistics(self, split_results: Dict[str, List[Dict]]):
        """Save comprehensive statistics about the fixed generation process."""
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
                "config": self.current_session.config,
                "version": "v2.2_fixed"
            },
            "generation": {
                "total_examples": sum(len(examples) for examples in split_results.values()),
                "split_breakdown": {
                    split: len(examples) for split, examples in split_results.items()
                },
                "quality_stats": self.global_stats,
                "agent_distribution": self._calculate_agent_distribution(split_results),
                "model_usage": self._calculate_model_usage_stats(split_results)
            },
            "batch_processing": {
                "strategy": "fixed_one_batch_per_split",
                "max_batch_size": self.config.max_batch_size,
                "total_batches_created": len(split_results),  # One batch per split (mostly)
                "model_consistency": "enforced",
                "batch_details": self._get_batch_details(split_results)
            },
            "connection_monitoring": {
                "monitoring_duration": connection_stats.get("last_check", 0) - self.current_session.start_time if connection_stats.get("last_check") else 0,
                "connection_health": connection_stats.get("required_connections_healthy", False),
                "connection_details": connection_stats.get("connection_details", {})
            },
            "system_info": {
                "recovery_used": self.recovery_manager.get_recovery_context() is not None,
                "session_stats": session_stats,
                "fixes_applied": [
                    "one_batch_per_split_max_140k",
                    "consistent_model_per_batch",
                    "no_arbitrary_batch_splitting"
                ]
            }
        }
        
        # Save to session directory
        with open(stats_file, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        
        # Also save to output directory for easy access
        output_stats_file = Path(self.config.output_dir) / "generation_stats.json"
        with open(output_stats_file, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        
        print(f"📊 Fixed comprehensive statistics saved:")
        print(f"   Session: {stats_file}")
        print(f"   Output: {output_stats_file}")
    
    def _calculate_model_usage_stats(self, split_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate detailed model usage statistics."""
        model_usage = {}
        
        for split_name, examples in split_results.items():
            if examples:
                # Get model info from first example (all should be the same)
                metadata = examples[0].get("_metadata", {})
                provider = metadata.get("provider", "unknown")
                model = metadata.get("model", "unknown")
                
                model_key = f"{provider}/{model}"
                if model_key not in model_usage:
                    model_usage[model_key] = {
                        "provider": provider,
                        "model": model,
                        "splits": [],
                        "total_examples": 0
                    }
                
                model_usage[model_key]["splits"].append({
                    "split": split_name,
                    "examples": len(examples)
                })
                model_usage[model_key]["total_examples"] += len(examples)
        
        return model_usage
    
    def _get_batch_details(self, split_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Get detailed batch information."""
        batch_details = []
        
        for split_name, examples in split_results.items():
            if examples:
                metadata = examples[0].get("_metadata", {})
                batch_details.append({
                    "split": split_name,
                    "batch_id": metadata.get("batch_id", "unknown"),
                    "provider": metadata.get("provider", "unknown"),
                    "model": metadata.get("model", "unknown"),
                    "examples": len(examples),
                    "consistent_model": True  # Enforced by fixed system
                })
        
        return batch_details
    
    def _calculate_agent_distribution(self, split_results: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Calculate distribution of examples by agent type."""
        agent_counts = {}
        
        for split_examples in split_results.values():
            for example in split_examples:
                agent_type = example.get("_metadata", {}).get("agent_type", "unknown")
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        return agent_counts
    
    def _get_total_target(self) -> int:
        """Get total target size for dataset."""
        return self.config.train_size + self.config.val_size + self.config.test_size
    
    # All other methods remain the same as EnhancedAgenticDataGenerator
    async def cleanup_session(self, cleanup_files: bool = False):
        """Clean up the current session."""
        if not self.current_session:
            return
        
        print(f"🧹 Cleaning up session: {self.current_session.session_id}")
        
        # Clean up batch tracking
        self.batch_tracker.cleanup_session(self.current_session.session_id, remove_files=False)
        
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
            "split_progress": self.split_coordinator.get_all_progress(),
            "version": "v2.2_fixed"
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
