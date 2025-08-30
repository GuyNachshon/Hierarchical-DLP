#!/usr/bin/env python3
"""
LLM API Integration for DLP Labeling

Provides unified interface for both OpenAI and Anthropic APIs
with proper rate limiting, error handling, and cost management.
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Optional imports - will provide helpful error messages if missing
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from llm_label_strategy import DLPPromptStrategy, LLMLabelingConfig

logger = logging.getLogger(__name__)

@dataclass
class LabelingResult:
    """Result from LLM labeling attempt."""
    success: bool
    labels: Optional[Dict[str, float]] = None
    reasoning: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    cost_estimate: float = 0.0
    response_time: float = 0.0

class LLMAPIClient:
    """Unified client for OpenAI and Anthropic APIs."""
    
    def __init__(self, config: LLMLabelingConfig):
        self.config = config
        self.prompt_strategy = DLPPromptStrategy()
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        
        self._setup_api_clients()
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
        self.error_count = 0
        
    def _setup_api_clients(self):
        """Initialize API clients based on configuration."""
        if self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized with model: {self.config.model}")
            
        elif self.config.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic client initialized with model: {self.config.model}")
            
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate API cost based on token counts."""
        # Rough cost estimates (as of 2024) - update as needed
        cost_per_1k = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
        
        if self.config.model not in cost_per_1k:
            return 0.001 * (prompt_tokens + completion_tokens) / 1000  # Rough estimate
        
        rates = cost_per_1k[self.config.model]
        cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1000
        return cost
    
    async def _call_openai(self, prompt: str) -> Tuple[Optional[str], float, str]:
        """Call OpenAI API with error handling."""
        try:
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.prompt_strategy.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            response_time = time.time() - start_time
            
            # Extract response and calculate cost
            content = response.choices[0].message.content
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            
            return content, cost, f"OpenAI call successful ({prompt_tokens}+{completion_tokens} tokens)"
            
        except Exception as e:
            return None, 0.0, f"OpenAI API error: {str(e)}"
    
    async def _call_anthropic(self, prompt: str) -> Tuple[Optional[str], float, str]:
        """Call Anthropic API with error handling."""
        try:
            start_time = time.time()
            
            # Combine system prompt and user prompt for Anthropic
            full_prompt = f"{self.prompt_strategy.system_prompt}\n\n{prompt}"
            
            response = self.anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            response_time = time.time() - start_time
            
            # Extract response and estimate cost
            content = response.content[0].text
            
            # Rough token estimation for cost calculation
            prompt_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
            completion_tokens = len(content.split()) * 1.3
            cost = self._estimate_cost(int(prompt_tokens), int(completion_tokens))
            
            return content, cost, f"Anthropic call successful (estimated {prompt_tokens:.0f}+{completion_tokens:.0f} tokens)"
            
        except Exception as e:
            return None, 0.0, f"Anthropic API error: {str(e)}"
    
    async def label_single_email(self, email_data: Dict[str, Any]) -> LabelingResult:
        """Label a single email using the configured LLM."""
        start_time = time.time()
        
        try:
            # Generate prompt
            prompt = self.prompt_strategy.create_labeling_prompt(email_data)
            
            # Call appropriate API
            if self.config.provider == "openai":
                response, cost, status = await self._call_openai(prompt)
            else:
                response, cost, status = await self._call_anthropic(prompt)
            
            response_time = time.time() - start_time
            self.request_count += 1
            self.total_cost += cost
            
            if response is None:
                self.error_count += 1
                return LabelingResult(
                    success=False,
                    error=status,
                    cost_estimate=cost,
                    response_time=response_time
                )
            
            # Parse response
            parsed_result = self.prompt_strategy.parse_llm_response(response)
            
            if parsed_result is None:
                self.error_count += 1
                return LabelingResult(
                    success=False,
                    error=f"Failed to parse LLM response: {response[:200]}...",
                    cost_estimate=cost,
                    response_time=response_time
                )
            
            # Extract reasoning if available
            reasoning = None
            try:
                # Try to extract reasoning from response
                response_json = json.loads(response.strip())
                if "reasoning" in response_json:
                    reasoning = response_json["reasoning"]
            except:
                pass  # Reasoning is optional
            
            return LabelingResult(
                success=True,
                labels=parsed_result,
                reasoning=reasoning,
                cost_estimate=cost,
                response_time=response_time
            )
            
        except Exception as e:
            self.error_count += 1
            return LabelingResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def label_batch(self, email_batch: List[Dict[str, Any]], 
                         progress_callback: Optional[callable] = None) -> List[LabelingResult]:
        """Label a batch of emails with rate limiting."""
        results = []
        
        for i, email_data in enumerate(email_batch):
            # Rate limiting
            if i > 0:
                await asyncio.sleep(self.config.rate_limit_delay)
            
            # Label email
            result = await self.label_single_email(email_data)
            results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(email_batch), result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                logger.info(f"Processed {i+1}/{len(email_batch)} emails. Success rate: {success_rate:.1f}%")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get labeling statistics."""
        success_rate = ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.request_count - self.error_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.total_cost / max(self.request_count, 1)
        }

def test_api_integration():
    """Test the API integration with a sample email."""
    print("🧪 Testing LLM API Integration")
    print("=" * 50)
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"OpenAI API Key: {'✅ Found' if openai_key else '❌ Not found'}")
    print(f"Anthropic API Key: {'✅ Found' if anthropic_key else '❌ Not found'}")
    
    if not openai_key and not anthropic_key:
        print("\n⚠️  No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Choose provider based on available keys
    if openai_key and OPENAI_AVAILABLE:
        config = LLMLabelingConfig(provider="openai", model="gpt-4o-mini")  # Cheaper for testing
    elif anthropic_key and ANTHROPIC_AVAILABLE:
        config = LLMLabelingConfig(provider="anthropic", model="claude-3-haiku-20240307")  # Cheaper for testing
    else:
        print("❌ No usable API client available")
        return
    
    print(f"\n🤖 Using {config.provider} with model {config.model}")
    
    # Sample email for testing
    test_email = {
        "user": {"role": "INTERN", "dept": "MARKETING"},
        "recipients": ["friend@gmail.com"],
        "subject": "Check out this salary data!",
        "body": "Found some interesting financial info. Everyone's salary is in this spreadsheet!",
        "attachments": [{"name": "salaries.xlsx", "size": 1024000, "mime": "application/vnd.ms-excel"}]
    }
    
    async def run_test():
        client = LLMAPIClient(config)
        
        print(f"\n📧 Testing with sample email:")
        print(json.dumps(test_email, indent=2))
        
        print(f"\n🔄 Calling {config.provider} API...")
        result = await client.label_single_email(test_email)
        
        print(f"\n📊 Results:")
        print(f"   Success: {result.success}")
        print(f"   Cost: ${result.cost_estimate:.4f}")
        print(f"   Response time: {result.response_time:.2f}s")
        
        if result.success:
            print(f"   Labels: {result.labels}")
            if result.reasoning:
                print(f"   Reasoning: {json.dumps(result.reasoning, indent=4)}")
        else:
            print(f"   Error: {result.error}")
        
        # Show statistics
        stats = client.get_statistics()
        print(f"\n📈 Statistics: {stats}")
    
    # Run the async test
    try:
        asyncio.run(run_test())
        print(f"\n✅ API integration test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    test_api_integration()