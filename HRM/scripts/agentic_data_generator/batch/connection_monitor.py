"""
Connection Monitor - Detects and handles network/API connection failures.

Provides robust connection monitoring with automatic detection of:
- Network connectivity issues
- API rate limiting and service unavailability  
- Authentication failures
- Timeout conditions
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ConnectionTest:
    """Configuration for a connection test."""
    name: str
    test_func: Callable
    timeout: float = 10.0
    required: bool = True  # Whether this connection is required for operation


@dataclass 
class ConnectionResult:
    """Result of a connection test."""
    name: str
    status: ConnectionStatus
    response_time: float
    error_message: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ConnectionMonitor:
    """Monitors API and network connections for batch processing."""
    
    def __init__(self):
        self.connection_tests: Dict[str, ConnectionTest] = {}
        self.last_results: Dict[str, ConnectionResult] = {}
        self.monitoring_active = False
        self.failure_callbacks: List[Callable[[Dict[str, ConnectionResult]], None]] = []
        self.recovery_callbacks: List[Callable[[Dict[str, ConnectionResult]], None]] = []
        
        # Configuration
        self.check_interval = 30.0  # seconds
        self.failure_threshold = 3  # consecutive failures before triggering callback
        self.consecutive_failures: Dict[str, int] = {}
        
        # Setup default tests
        self._setup_default_tests()
    
    def _setup_default_tests(self):
        """Setup default connection tests for common APIs."""
        
        # OpenAI API test
        self.add_connection_test(
            name="openai_api",
            test_func=self._test_openai_connection,
            timeout=15.0,
            required=True
        )
        
        # Anthropic API test  
        self.add_connection_test(
            name="anthropic_api", 
            test_func=self._test_anthropic_connection,
            timeout=15.0,
            required=True
        )
        
        # General internet connectivity
        self.add_connection_test(
            name="internet_connectivity",
            test_func=self._test_internet_connection,
            timeout=10.0,
            required=True
        )
    
    async def _test_openai_connection(self) -> ConnectionResult:
        """Test OpenAI API connectivity."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return ConnectionResult(
                name="openai_api",
                status=ConnectionStatus.FAILED,
                response_time=0.0,
                error_message="OPENAI_API_KEY not set"
            )
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Test with a minimal models list request
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15.0)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return ConnectionResult(
                            name="openai_api",
                            status=ConnectionStatus.HEALTHY,
                            response_time=response_time
                        )
                    elif response.status == 429:
                        return ConnectionResult(
                            name="openai_api", 
                            status=ConnectionStatus.DEGRADED,
                            response_time=response_time,
                            error_message=f"Rate limited (HTTP {response.status})"
                        )
                    else:
                        return ConnectionResult(
                            name="openai_api",
                            status=ConnectionStatus.FAILED,
                            response_time=response_time,
                            error_message=f"HTTP {response.status}"
                        )
        
        except asyncio.TimeoutError:
            return ConnectionResult(
                name="openai_api",
                status=ConnectionStatus.FAILED,
                response_time=time.time() - start_time,
                error_message="Timeout"
            )
        except Exception as e:
            return ConnectionResult(
                name="openai_api",
                status=ConnectionStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_anthropic_connection(self) -> ConnectionResult:
        """Test Anthropic API connectivity."""
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            return ConnectionResult(
                name="anthropic_api",
                status=ConnectionStatus.FAILED,
                response_time=0.0,
                error_message="ANTHROPIC_API_KEY not set"
            )
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                
                # Test with a minimal message request
                test_data = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
                
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=test_data,
                    timeout=aiohttp.ClientTimeout(total=15.0)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return ConnectionResult(
                            name="anthropic_api",
                            status=ConnectionStatus.HEALTHY,
                            response_time=response_time
                        )
                    elif response.status == 429:
                        return ConnectionResult(
                            name="anthropic_api",
                            status=ConnectionStatus.DEGRADED,
                            response_time=response_time,
                            error_message=f"Rate limited (HTTP {response.status})"
                        )
                    else:
                        return ConnectionResult(
                            name="anthropic_api",
                            status=ConnectionStatus.FAILED,
                            response_time=response_time,
                            error_message=f"HTTP {response.status}"
                        )
        
        except asyncio.TimeoutError:
            return ConnectionResult(
                name="anthropic_api",
                status=ConnectionStatus.FAILED,
                response_time=time.time() - start_time,
                error_message="Timeout"
            )
        except Exception as e:
            return ConnectionResult(
                name="anthropic_api",
                status=ConnectionStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_internet_connection(self) -> ConnectionResult:
        """Test basic internet connectivity."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test multiple reliable endpoints
                test_urls = [
                    "https://www.google.com",
                    "https://www.cloudflare.com",
                    "https://httpbin.org/get"
                ]
                
                for url in test_urls:
                    try:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=10.0)
                        ) as response:
                            if response.status == 200:
                                return ConnectionResult(
                                    name="internet_connectivity",
                                    status=ConnectionStatus.HEALTHY,
                                    response_time=time.time() - start_time
                                )
                    except:
                        continue
                
                # If all URLs failed
                return ConnectionResult(
                    name="internet_connectivity",
                    status=ConnectionStatus.FAILED,
                    response_time=time.time() - start_time,
                    error_message="All test URLs failed"
                )
        
        except Exception as e:
            return ConnectionResult(
                name="internet_connectivity",
                status=ConnectionStatus.FAILED,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def add_connection_test(self, name: str, test_func: Callable, timeout: float = 10.0, required: bool = True):
        """Add a custom connection test."""
        self.connection_tests[name] = ConnectionTest(
            name=name,
            test_func=test_func,
            timeout=timeout,
            required=required
        )
        self.consecutive_failures[name] = 0
    
    def add_failure_callback(self, callback: Callable[[Dict[str, ConnectionResult]], None]):
        """Add callback to be called when connections fail."""
        self.failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[Dict[str, ConnectionResult]], None]):
        """Add callback to be called when connections recover."""
        self.recovery_callbacks.append(callback)
    
    async def check_all_connections(self) -> Dict[str, ConnectionResult]:
        """Run all connection tests and return results."""
        results = {}
        
        # Run all tests concurrently
        tasks = []
        for name, test in self.connection_tests.items():
            task = asyncio.create_task(
                asyncio.wait_for(test.test_func(), timeout=test.timeout)
            )
            tasks.append((name, task))
        
        # Collect results
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except asyncio.TimeoutError:
                results[name] = ConnectionResult(
                    name=name,
                    status=ConnectionStatus.FAILED,
                    response_time=self.connection_tests[name].timeout,
                    error_message="Test timeout"
                )
            except Exception as e:
                results[name] = ConnectionResult(
                    name=name,
                    status=ConnectionStatus.FAILED,
                    response_time=0.0,
                    error_message=f"Test error: {e}"
                )
        
        # Update consecutive failure counts
        self._update_failure_counts(results)
        
        # Store results
        self.last_results = results
        
        return results
    
    def _update_failure_counts(self, results: Dict[str, ConnectionResult]):
        """Update consecutive failure counts and trigger callbacks."""
        previously_failed = set()
        currently_failed = set()
        
        # Track which connections were previously failed
        for name, last_result in self.last_results.items():
            if (self.connection_tests.get(name, ConnectionTest("", None)).required and 
                last_result.status == ConnectionStatus.FAILED):
                previously_failed.add(name)
        
        # Update failure counts
        for name, result in results.items():
            test = self.connection_tests.get(name)
            if not test or not test.required:
                continue
            
            if result.status == ConnectionStatus.FAILED:
                self.consecutive_failures[name] += 1
                currently_failed.add(name)
                
                # Trigger failure callback if threshold reached
                if (self.consecutive_failures[name] == self.failure_threshold and 
                    name not in previously_failed):
                    print(f"🚨 Connection failed: {name} ({result.error_message})")
                    for callback in self.failure_callbacks:
                        try:
                            callback(results)
                        except Exception as e:
                            print(f"⚠️  Failure callback error: {e}")
            else:
                # Connection recovered
                if self.consecutive_failures[name] > 0:
                    if name in previously_failed:
                        print(f"🟢 Connection recovered: {name}")
                        for callback in self.recovery_callbacks:
                            try:
                                callback(results)
                            except Exception as e:
                                print(f"⚠️  Recovery callback error: {e}")
                    
                    self.consecutive_failures[name] = 0
    
    async def start_monitoring(self):
        """Start continuous connection monitoring."""
        self.monitoring_active = True
        print(f"📡 Starting connection monitoring (check interval: {self.check_interval}s)")
        
        while self.monitoring_active:
            try:
                await self.check_all_connections()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️  Connection monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    def stop_monitoring(self):
        """Stop connection monitoring."""
        self.monitoring_active = False
        print("🔌 Connection monitoring stopped")
    
    def get_connection_status(self) -> Dict[str, ConnectionStatus]:
        """Get current connection status for all tests."""
        return {name: result.status for name, result in self.last_results.items()}
    
    def are_required_connections_healthy(self) -> bool:
        """Check if all required connections are healthy."""
        for name, test in self.connection_tests.items():
            if test.required:
                result = self.last_results.get(name)
                if not result or result.status == ConnectionStatus.FAILED:
                    return False
        return True
    
    def get_failed_connections(self) -> List[str]:
        """Get list of currently failed required connections."""
        failed = []
        for name, test in self.connection_tests.items():
            if test.required:
                result = self.last_results.get(name)
                if result and result.status == ConnectionStatus.FAILED:
                    failed.append(name)
        return failed
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if not self.last_results:
            return {"status": "no_data", "last_check": None}
        
        latest_timestamp = max(result.timestamp for result in self.last_results.values())
        
        stats = {
            "status": "monitoring" if self.monitoring_active else "stopped",
            "last_check": latest_timestamp,
            "total_tests": len(self.connection_tests),
            "healthy_connections": len([r for r in self.last_results.values() if r.status == ConnectionStatus.HEALTHY]),
            "failed_connections": len([r for r in self.last_results.values() if r.status == ConnectionStatus.FAILED]),
            "degraded_connections": len([r for r in self.last_results.values() if r.status == ConnectionStatus.DEGRADED]),
            "required_connections_healthy": self.are_required_connections_healthy(),
            "connection_details": {
                name: {
                    "status": result.status.value,
                    "response_time": result.response_time,
                    "error": result.error_message,
                    "consecutive_failures": self.consecutive_failures.get(name, 0)
                }
                for name, result in self.last_results.items()
            }
        }
        
        return stats