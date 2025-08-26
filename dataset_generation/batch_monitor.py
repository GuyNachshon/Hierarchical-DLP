#!/usr/bin/env python3
"""
Thread-based Batch Monitor for efficient batch API monitoring.

Handles:
- Continuous monitoring of active batches in separate thread
- Automatic result retrieval when batches complete
- Integration with data splits (train/val/test)
- Enhanced error handling and timeout management
"""

import threading
import time
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import logging

from batch_tracker import BatchTracker, BatchStatus, BatchMetadata


@dataclass
class BatchResult:
    """Result from a completed batch"""
    batch_id: str
    provider: str
    model: str
    results: List[Optional[str]]
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchCompletionCallback:
    """Callback for batch completion"""
    batch_id: str
    callback: Callable[[BatchResult], None]
    split_name: str


class BatchMonitor:
    """Thread-based batch monitor for continuous batch tracking"""
    
    def __init__(self, 
                 batch_tracker: BatchTracker,
                 clients: Dict[str, Any],
                 check_interval: int = 30,
                 max_retries: int = 3):
        self.batch_tracker = batch_tracker
        self.clients = clients
        self.check_interval = check_interval
        self.max_retries = max_retries
        
        # Threading components
        self.monitor_thread = None
        self.shutdown_event = threading.Event()
        self.completion_queue = Queue()
        
        # Callback management
        self.completion_callbacks: Dict[str, BatchCompletionCallback] = {}
        self.callback_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "batches_monitored": 0,
            "batches_completed": 0,
            "batches_failed": 0,
            "total_monitoring_time": 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start the batch monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Batch monitor already running")
            return
            
        self.shutdown_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="BatchMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("🚀 Batch monitor thread started")
        
        # Start API fetching in separate background thread to avoid blocking
        api_fetch_thread = threading.Thread(
            target=self._fetch_completed_batches_from_api,
            name="ApiFetch",
            daemon=True
        )
        api_fetch_thread.start()
        
    def stop_monitoring(self, timeout: int = 30):
        """Stop the batch monitoring thread gracefully"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            return
            
        self.logger.info("🛑 Stopping batch monitor...")
        self.shutdown_event.set()
        self.monitor_thread.join(timeout=timeout)
        
        if self.monitor_thread.is_alive():
            self.logger.warning("⚠️ Batch monitor did not stop gracefully")
        else:
            self.logger.info("✅ Batch monitor stopped")
    
    def register_batch_completion(self, 
                                batch_id: str, 
                                callback: Callable[[BatchResult], None],
                                split_name: str):
        """Register a callback for when a batch completes"""
        with self.callback_lock:
            self.completion_callbacks[batch_id] = BatchCompletionCallback(
                batch_id=batch_id,
                callback=callback,
                split_name=split_name
            )
        self.logger.info(f"📝 Registered completion callback for batch {batch_id[:8]}... (split: {split_name})")
    
    def unregister_batch_completion(self, batch_id: str):
        """Unregister a batch completion callback"""
        with self.callback_lock:
            if batch_id in self.completion_callbacks:
                del self.completion_callbacks[batch_id]
                self.logger.info(f"🗑️ Unregistered callback for batch {batch_id[:8]}...")
    
    def get_completion_result(self, timeout: Optional[float] = None) -> Optional[BatchResult]:
        """Get next completion result from queue (non-blocking)"""
        try:
            return self.completion_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        self.logger.info("🔄 Batch monitor loop started")
        start_time = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Get active batches from tracker
                active_batches = self.batch_tracker.get_active_batches()
                
                if active_batches:
                    self.logger.debug(f"📊 Monitoring {len(active_batches)} active batches")
                    
                    # Check each active batch
                    for batch_id, metadata in active_batches.items():
                        if self.shutdown_event.is_set():
                            break
                            
                        try:
                            self._check_batch_status(batch_id, metadata)
                        except Exception as e:
                            self.logger.error(f"❌ Error checking batch {batch_id[:8]}...: {e}")
                            # Mark as failed after max retries
                            self._handle_batch_failure(batch_id, str(e))
                
                # Clean up timed-out batches
                expired_batch_ids = self.batch_tracker.cleanup_timed_out_batches()
                for batch_id in expired_batch_ids:
                    self._handle_batch_failure(batch_id, "Batch timed out")
                
                # Check for failed batches that need retry with smaller sizes
                self._check_failed_batches_for_retry()
                
                # Wait for next check interval
                self.shutdown_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitor loop: {e}")
                # Continue monitoring even if there's an error
                self.shutdown_event.wait(5)  # Short delay before retrying
        
        # Update stats
        self.stats["total_monitoring_time"] = time.time() - start_time
        self.logger.info("🏁 Batch monitor loop ended")
    
    def _check_batch_status(self, batch_id: str, metadata: BatchMetadata):
        """Check status of a specific batch"""
        try:
            if metadata.provider == "openai":
                self._check_openai_batch(batch_id, metadata)
            elif metadata.provider == "anthropic":
                self._check_anthropic_batch(batch_id, metadata)
            else:
                raise ValueError(f"Unknown provider: {metadata.provider}")
        except Exception as e:
            self.logger.error(f"❌ Failed to check {metadata.provider} batch {batch_id[:8]}...: {e}")
            raise
    
    def _check_openai_batch(self, batch_id: str, metadata: BatchMetadata):
        """Check OpenAI batch status"""
        client = self.clients.get("openai")
        if not client:
            raise ValueError("OpenAI client not available")
        
        # Retrieve batch status
        batch = client.batches.retrieve(batch_id)
        
        # Update status in tracker
        if batch.status != metadata.status:
            self.logger.info(f"📊 OpenAI batch {batch_id[:8]}... status: {metadata.status} → {batch.status}")
            
            if batch.status == "validating":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.VALIDATING)
            elif batch.status == "in_progress":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.IN_PROGRESS)
            elif batch.status == "finalizing":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.FINALIZING)
            elif batch.status == "completed":
                self._handle_openai_batch_completion(batch_id, batch, metadata)
            elif batch.status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch status: {batch.status}"
                if hasattr(batch, 'errors') and batch.errors:
                    error_msg += f", errors: {batch.errors}"
                self._handle_batch_failure(batch_id, error_msg)
    
    def _check_anthropic_batch(self, batch_id: str, metadata: BatchMetadata):
        """Check Anthropic batch status"""
        client = self.clients.get("anthropic")
        if not client:
            raise ValueError("Anthropic client not available")
        
        # Retrieve batch status
        batch = client.messages.batches.retrieve(batch_id)
        
        # Update status in tracker
        if batch.processing_status != metadata.status:
            self.logger.info(f"📊 Anthropic batch {batch_id[:8]}... status: {metadata.status} → {batch.processing_status}")
            
            if batch.processing_status == "validating":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.VALIDATING)
            elif batch.processing_status == "in_progress":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.IN_PROGRESS)
            elif batch.processing_status == "finalizing":
                self.batch_tracker.update_batch_status(batch_id, BatchStatus.FINALIZING)
            elif batch.processing_status == "completed":
                self._handle_anthropic_batch_completion(batch_id, batch, metadata)
            elif batch.processing_status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch status: {batch.processing_status}"
                self._handle_batch_failure(batch_id, error_msg)
    
    def _handle_openai_batch_completion(self, batch_id: str, batch, metadata: BatchMetadata):
        """Handle completion of an OpenAI batch"""
        try:
            self.logger.info(f"✅ OpenAI batch {batch_id[:8]}... completed, retrieving results...")
            
            # Download and process results
            result_file_id = batch.output_file_id
            if not result_file_id:
                raise ValueError("No output file ID in completed batch")
            
            # Download results file
            result_file = self.clients["openai"].files.content(result_file_id)
            results_content = result_file.content.decode('utf-8')
            
            # Parse results and map to original order
            results = self._parse_openai_results(results_content, metadata.request_indices)
            
            # Create result object
            batch_result = BatchResult(
                batch_id=batch_id,
                provider="openai",
                model=metadata.model,
                results=results,
                success=True
            )
            
            # Mark as completed and trigger callback
            self.batch_tracker.complete_batch(batch_id, success=True)
            self._trigger_completion_callback(batch_result)
            self.stats["batches_completed"] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Failed to handle OpenAI batch completion {batch_id[:8]}...: {e}")
            self._handle_batch_failure(batch_id, str(e))
    
    def _handle_anthropic_batch_completion(self, batch_id: str, batch, metadata: BatchMetadata):
        """Handle completion of an Anthropic batch"""
        try:
            self.logger.info(f"✅ Anthropic batch {batch_id[:8]}... completed, retrieving results...")
            
            # Get results from batch
            results = self._parse_anthropic_results(batch, metadata.request_indices)
            
            # Create result object
            batch_result = BatchResult(
                batch_id=batch_id,
                provider="anthropic", 
                model=metadata.model,
                results=results,
                success=True
            )
            
            # Mark as completed and trigger callback
            self.batch_tracker.complete_batch(batch_id, success=True)
            self._trigger_completion_callback(batch_result)
            self.stats["batches_completed"] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Failed to handle Anthropic batch completion {batch_id[:8]}...: {e}")
            self._handle_batch_failure(batch_id, str(e))
    
    def _parse_openai_results(self, results_content: str, request_indices: List[int]) -> List[Optional[str]]:
        """Parse OpenAI batch results and map to original order"""
        # Initialize results array
        results = [None] * len(request_indices)
        
        # Parse each result line
        for line in results_content.strip().split('\n'):
            if not line:
                continue
                
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if custom_id and custom_id.startswith("request-"):
                    # Extract index from custom_id
                    index = int(custom_id.split("-")[1])
                    
                    if "response" in result and "body" in result["response"]:
                        # Extract content from response
                        response_body = result["response"]["body"]
                        if "choices" in response_body and len(response_body["choices"]) > 0:
                            choice = response_body["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                results[index] = choice["message"]["content"]
                    elif "error" in result:
                        self.logger.warning(f"⚠️ Error in request {custom_id}: {result['error']}")
                        
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"⚠️ Failed to parse result line: {e}")
        
        return results
    
    def _parse_anthropic_results(self, batch, request_indices: List[int]) -> List[Optional[str]]:
        """Parse Anthropic batch results and map to original order"""
        # Initialize results array
        results = [None] * len(request_indices)
        
        # Process each result in the batch
        if hasattr(batch, 'results') and batch.results:
            for result in batch.results:
                try:
                    custom_id = result.custom_id
                    
                    if custom_id and custom_id.startswith("request-"):
                        # Extract index from custom_id
                        index = int(custom_id.split("-")[1])
                        
                        if result.result and result.result.type == "message":
                            # Extract content from message
                            message = result.result.message
                            if message.content and len(message.content) > 0:
                                content_block = message.content[0]
                                if hasattr(content_block, 'text'):
                                    results[index] = content_block.text
                        elif result.result and result.result.type == "error":
                            self.logger.warning(f"⚠️ Error in request {custom_id}: {result.result.error}")
                            
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"⚠️ Failed to parse result: {e}")
        
        return results
    
    def _handle_batch_failure(self, batch_id: str, error_message: str):
        """Handle batch failure"""
        self.logger.error(f"❌ Batch {batch_id[:8]}... failed: {error_message}")
        
        # Get metadata for creating result
        metadata = self.batch_tracker.active_batches.get(batch_id)
        if metadata:
            # Create failed result
            batch_result = BatchResult(
                batch_id=batch_id,
                provider=metadata.provider,
                model=metadata.model,
                results=[None] * metadata.request_count,
                success=False,
                error_message=error_message
            )
            
            # Mark as failed and trigger callback
            self.batch_tracker.complete_batch(batch_id, success=False, error_message=error_message)
            self._trigger_completion_callback(batch_result)
        
        self.stats["batches_failed"] += 1
    
    def _trigger_completion_callback(self, batch_result: BatchResult):
        """Trigger completion callback for a batch"""
        with self.callback_lock:
            callback_info = self.completion_callbacks.get(batch_result.batch_id)
            if callback_info:
                try:
                    # Add result to queue for main thread processing
                    self.completion_queue.put(batch_result)
                    
                    # Also call the callback directly if provided
                    if callback_info.callback:
                        callback_info.callback(batch_result)
                        
                    self.logger.info(f"🎯 Triggered callback for batch {batch_result.batch_id[:8]}... (split: {callback_info.split_name})")
                    
                    # Clean up callback
                    del self.completion_callbacks[batch_result.batch_id]
                    
                except Exception as e:
                    self.logger.error(f"❌ Error triggering callback for batch {batch_result.batch_id[:8]}...: {e}")
    
    def _check_failed_batches_for_retry(self):
        """Check failed batches and queue them for retry with smaller sizes"""
        from task_dashboard import get_task_manager, TaskStatus
        task_manager = get_task_manager()
        
        # Get recent failed batches that haven't been retried yet
        recovery_info = self.batch_tracker.get_recovery_info()
        
        # Use dashboard updates instead of print spam - only update every 30 seconds
        current_time = time.time()
        if not hasattr(self, '_last_recovery_update'):
            self._last_recovery_update = 0
            
        if current_time - self._last_recovery_update >= 30:  # Update every 30s instead of every 1s
            if recovery_info["active_batches"] > 0:
                task_manager.update_thread("BatchMonitor", "monitoring", 
                                         f"Monitoring {recovery_info['active_batches']} active batches, "
                                         f"{recovery_info['total_pending_requests']} pending requests")
            
            # Check for timed out batches that need recovery
            timed_out_count = recovery_info["timed_out_batches"]
            if timed_out_count > 0:
                task_manager.update_thread("BatchMonitor", "warning", 
                                         f"{timed_out_count} batches timed out - recovery needed")
                self.logger.debug(f"⚠️  {timed_out_count} batches have timed out and may need recovery")
                
            self._last_recovery_update = current_time
            
        # Force cleanup of very old batches (>1 hour)
        self._cleanup_stuck_batches()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        active_count = len(self.batch_tracker.get_active_batches())
        return {
            **self.stats,
            "active_batches": active_count,
            "pending_callbacks": len(self.completion_callbacks),
            "is_monitoring": self.monitor_thread and self.monitor_thread.is_alive()
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
    
    def _fetch_completed_batches_from_api(self):
        """Fetch completed batches from API providers that may have finished while offline"""
        self.logger.info("🔍 Fetching completed batches from API providers...")
        
        # Fetch from OpenAI
        self._fetch_openai_completed_batches()
        
        # Note: Anthropic doesn't support listing batches, so we can't fetch from there
        self.logger.info("ℹ️ Anthropic doesn't support listing batches - only active batches in tracker will be monitored")
    
    def _fetch_openai_completed_batches(self):
        """Fetch completed batches from OpenAI API"""
        openai_client = self.clients.get("openai")
        if not openai_client:
            self.logger.info("⚠️ OpenAI client not available, skipping completed batch fetch")
            return
        
        try:
            self.logger.info("📡 Fetching recent OpenAI batches...")
            
            # List recent batches (limit 100 to get recent ones)
            batches_response = openai_client.batches.list(limit=100)
            
            completed_count = 0
            recovered_count = 0
            
            # Handle both sync and async responses
            if hasattr(batches_response, 'data'):
                batches_data = batches_response.data
            else:
                # For AsyncPaginator, we need to iterate
                batches_data = list(batches_response)
            
            for batch in batches_data:
                batch_id = batch.id
                
                # Check if this batch is in our tracker
                if batch_id in self.batch_tracker.active_batches:
                    # Update status in tracker
                    if batch.status != self.batch_tracker.active_batches[batch_id].status:
                        self.logger.info(f"📊 Updating batch {batch_id[:8]}... status: {self.batch_tracker.active_batches[batch_id].status} → {batch.status}")
                        
                        if batch.status == "completed":
                            # Process completed batch
                            metadata = self.batch_tracker.active_batches[batch_id]
                            self._handle_openai_batch_completion(batch_id, batch, metadata)
                            completed_count += 1
                        elif batch.status in ["failed", "expired", "cancelled"]:
                            # Handle failed batch
                            error_msg = f"Batch status: {batch.status}"
                            self._handle_batch_failure(batch_id, error_msg)
                            completed_count += 1
                        else:
                            # Update status for in-progress batches
                            self.batch_tracker.update_batch_status(batch_id, BatchStatus(batch.status))
                
                elif batch.status == "completed":
                    # Check if batch is already in our completed list
                    if batch_id in self.batch_tracker.completed_batches:
                        # Already processed, skip
                        continue
                        
                    # Found a completed batch not in our tracker - recover it
                    self.logger.info(f"🚀 Found orphaned completed batch: {batch_id[:8]}...")
                    
                    # Try to recover this batch
                    try:
                        # Create minimal metadata for recovery
                        request_count = getattr(batch.request_counts, 'total', 0) if hasattr(batch, 'request_counts') else 0
                        
                        # Register in tracker as completed
                        metadata = self.batch_tracker.create_batch(
                            batch_id=batch_id,
                            provider="openai",
                            model="unknown_recovered",  # We don't know the original model
                            request_count=request_count,
                            request_indices=list(range(request_count))
                        )
                        
                        # Process the completed batch
                        self._handle_openai_batch_completion(batch_id, batch, metadata)
                        recovered_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"❌ Failed to recover batch {batch_id[:8]}...: {e}")
            
            if completed_count > 0 or recovered_count > 0:
                self.logger.info(f"✅ Processed {completed_count} completed tracked batches, recovered {recovered_count} orphaned batches")
            else:
                self.logger.info("ℹ️ No completed batches found")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch OpenAI batches: {e}")
    
    def _fetch_anthropic_completed_batches(self):
        """Fetch completed batches from Anthropic API"""
        anthropic_client = self.clients.get("anthropic")
        if not anthropic_client:
            self.logger.info("⚠️ Anthropic client not available, skipping completed batch fetch")
            return
        
        try:
            self.logger.info("📡 Fetching recent Anthropic batches...")
            
            # Note: As of current API documentation, Anthropic doesn't provide a list batches endpoint
            # We can only monitor batches we already know about from the tracker
            self.logger.info("ℹ️ Anthropic API doesn't support listing batches - only monitoring tracked batches")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch Anthropic batches: {e}")
    
    def _cleanup_stuck_batches(self):
        """Force cleanup of batches older than 1 hour to prevent blocking"""
        STUCK_BATCH_TIMEOUT = 3600  # 1 hour in seconds
        
        active_batches = self.batch_tracker.get_active_batches()
        current_time = time.time()
        stuck_batches = []
        
        for batch_id, metadata in active_batches.items():
            batch_age = current_time - metadata.created_at
            if batch_age > STUCK_BATCH_TIMEOUT:
                stuck_batches.append((batch_id, batch_age))
        
        if stuck_batches:
            from task_dashboard import get_task_manager
            task_manager = get_task_manager()
            
            for batch_id, age in stuck_batches:
                self.logger.warning(f"🗑️ Force cleaning stuck batch {batch_id[:8]}... (age: {age:.0f}s)")
                task_manager.update_thread("BatchCleanup", "cleaning", f"Removing stuck batch {batch_id[:8]}... ({age:.0f}s old)")
                
                # Force completion as failed to clean up
                self.batch_tracker.complete_batch(batch_id, success=False, error_message=f"Force cleanup after {age:.0f}s timeout")
                
            task_manager.complete_task("batch_cleanup", f"Cleaned up {len(stuck_batches)} stuck batches")