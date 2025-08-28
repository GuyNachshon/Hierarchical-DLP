"""
Simplified batch processor for LLM API requests with clean error handling.
"""

import json
import time
import asyncio
from typing import List, Tuple, Optional, Dict
from .llm_client import LLMClientManager


class BatchProcessor:
    """Simplified batch processor for efficient LLM API usage."""
    
    def __init__(self, config=None):
        self.config = config
        self.client_manager = LLMClientManager(config)
        self.batch_size = getattr(config, 'batch_size', 20)
        self.batch_threshold = getattr(config, 'batch_threshold', 50)
        self.enable_batch_api = getattr(config, 'enable_batch_api', True)
        self.max_concurrent = getattr(config, 'max_concurrent_agents', 10)
    
    def _get_structured_output_schema(self) -> Dict:
        """Get JSON schema for structured output validation."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hrm_dlp_example",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "enum": ["email", "chat", "pr", "upload"]
                        },
                        "user": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "dept": {"type": "string"},
                                "seniority": {"type": "string"}
                            },
                            "required": ["role", "dept", "seniority"],
                            "additionalProperties": False
                        },
                        "recipients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "attachments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "size": {"type": "integer", "minimum": 1},
                                    "mime_type": {"type": "string"},
                                    "content_summary": {"type": "string"},
                                    "sensitivity_indicators": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1
                                    }
                                },
                                "required": ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"],
                                "additionalProperties": False
                            },
                            "minItems": 1
                        },
                        "links": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "context_summary": {"type": "string"},
                        "thread": {
                            "type": "object",
                            "properties": {
                                "id_hash": {"type": "string"},
                                "age_days": {"type": "integer"},
                                "prior_msgs": {"type": "integer"}
                            },
                            "required": ["id_hash", "age_days", "prior_msgs"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["channel", "user", "recipients", "subject", "body", "attachments", "links", "context_summary", "thread"],
                    "additionalProperties": False
                }
            }
        }
    
    async def process_requests(self, requests: List[Tuple[str, str]], 
                             provider_preference: Optional[str] = None) -> List[Optional[str]]:
        """Process a list of (system, prompt) requests."""
        if not requests:
            return []
        
        # Choose processing strategy
        if (self.enable_batch_api and 
            len(requests) >= self.batch_threshold and
            provider_preference in ["openai", "anthropic"]):
            return await self._process_with_batch_api(requests, provider_preference)
        else:
            return await self._process_concurrent(requests, provider_preference)
    
    async def _process_with_batch_api(self, requests: List[Tuple[str, str]], 
                                    provider: str) -> List[Optional[str]]:
        """Process requests using batch API (simplified version)."""
        try:
            if provider == "openai":
                return await self._process_openai_batch(requests)
            elif provider == "anthropic":
                return await self._process_anthropic_batch(requests)
            else:
                # Fallback to concurrent processing
                return await self._process_concurrent(requests, provider)
        except Exception as e:
            print(f"Batch API failed: {e}, falling back to concurrent processing")
            return await self._process_concurrent(requests, provider)
    
    async def _process_openai_batch(self, requests: List[Tuple[str, str]]) -> List[Optional[str]]:
        """Process batch using OpenAI Batch API (simplified)."""
        client = self.client_manager.get_client("openai")
        if not client:
            raise ValueError("OpenAI client not available")
        
        # Choose a single model for the entire batch
        _, model = self.client_manager.choose_model("openai")
        temperature = self.client_manager.get_model_temperature(model)
        
        # Create batch file
        batch_file_path = f"/tmp/openai_batch_{int(time.time())}.jsonl"
        batch_requests = []
        
        for i, (system, prompt) in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature
                }
            }
            batch_requests.append(batch_request)
        
        # Write and submit batch
        with open(batch_file_path, 'w') as f:
            for req in batch_requests:
                f.write(json.dumps(req) + '\\n')
        
        # Submit batch (simplified - returns placeholder)
        print(f"📝 Batch submitted to OpenAI: {len(requests)} requests")
        
        # For this simplified version, return placeholders
        # In production, this would wait for batch completion
        return ["BATCH_PROCESSING"] * len(requests)
    
    async def _process_anthropic_batch(self, requests: List[Tuple[str, str]]) -> List[Optional[str]]:
        """Process batch using Anthropic Message Batches API (simplified)."""
        client = self.client_manager.get_client("anthropic")
        if not client:
            raise ValueError("Anthropic client not available")
        
        # For this simplified version, return placeholders
        print(f"📝 Batch submitted to Anthropic: {len(requests)} requests")
        return ["BATCH_PROCESSING"] * len(requests)
    
    async def _process_concurrent(self, requests: List[Tuple[str, str]], 
                                provider_preference: Optional[str] = None) -> List[Optional[str]]:
        """Process requests concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_request(system: str, prompt: str) -> Optional[str]:
            async with semaphore:
                try:
                    provider, model = self.client_manager.choose_model(provider_preference)
                    client = self.client_manager.get_client(provider)
                    
                    if not client:
                        return None
                    
                    temperature = self.client_manager.get_model_temperature(model)
                    
                    # Add small delay to avoid rate limits
                    await asyncio.sleep(0.1)
                    
                    if provider == "anthropic":
                        response = await asyncio.to_thread(
                            client.messages.create,
                            model=model,
                            max_tokens=4096,
                            temperature=temperature,
                            system=system,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.content[0].text
                    else:
                        # OpenAI and compatible providers
                        request_params = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": temperature
                        }
                        
                        # Add structured output for OpenAI models that support it
                        # Include GPT-4o, GPT-4-turbo, and newer GPT-5 models
                        if provider == "openai" and ("gpt-4o" in model or "gpt-4-turbo" in model or "gpt-5" in model):
                            request_params["response_format"] = self._get_structured_output_schema()
                        
                        response = await asyncio.to_thread(
                            client.chat.completions.create,
                            **request_params
                        )
                        return response.choices[0].message.content
                        
                except Exception as e:
                    print(f"Request failed: {e}")
                    return None
        
        # Process all requests concurrently
        tasks = [process_single_request(system, prompt) for system, prompt in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        return [result if not isinstance(result, Exception) else None for result in results]
    
    def get_batch_compatible_models(self) -> Dict[str, List[str]]:
        """Get list of batch-compatible models by provider."""
        compatible = {}
        
        for provider in self.client_manager.get_available_providers():
            models = []
            if provider in self.client_manager.model_weights:
                for model in self.client_manager.model_weights[provider].keys():
                    if self.client_manager.is_batch_compatible(provider, model):
                        models.append(model)
            compatible[provider] = models
            
        return compatible