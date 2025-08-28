"""
Fixed Batch Processor - Ensures consistent model usage per batch.

Key fixes:
- Added process_requests_with_fixed_model() method
- Ensures all requests in a batch use the same model
- Maintains batch API support with consistent model selection
"""

import json
import time
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from .llm_client import LLMClientManager


class FixedBatchProcessor:
    """Fixed batch processor ensuring consistent model usage per batch."""
    
    def __init__(self, config=None):
        self.config = config
        self.client_manager = LLMClientManager(config)
        self.batch_size = getattr(config, 'batch_size', 20)
        self.batch_threshold = getattr(config, 'batch_threshold', 50)
        self.enable_batch_api = getattr(config, 'enable_batch_api', True)
        self.max_concurrent = getattr(config, 'max_concurrent_agents', 10)
    
    async def process_requests(self, requests: List[Tuple[str, str]], 
                             provider_preference: Optional[str] = None) -> List[Optional[str]]:
        """Process a list of (system, prompt) requests with model consistency."""
        if not requests:
            return []
        
        # Choose ONE model for the entire batch (only structured-output compatible models)
        provider, model = self.client_manager.choose_model(provider_preference)
        
        print(f"🤖 Processing {len(requests)} requests with consistent model: {provider}/{model}")
        
        # Verify structured output support
        if provider == "openai" and not self.client_manager.supports_structured_outputs(provider, model):
            raise ValueError(f"Model {model} does not support structured outputs required for batch processing")
        
        # Use the fixed model method
        return await self.process_requests_with_fixed_model(requests, provider, model)
    
    async def process_requests_with_fixed_model(self, requests: List[Tuple[str, str]], 
                                              provider: str, model: str) -> List[Optional[str]]:
        """Process requests using a specific fixed model for consistency."""
        if not requests:
            return []
        
        print(f"🔧 Fixed model processing: {len(requests)} requests using {provider}/{model}")
        
        # Choose processing strategy based on batch size and API support
        if (self.enable_batch_api and 
            len(requests) >= self.batch_threshold and
            provider in ["openai", "anthropic"]):
            return await self._process_with_batch_api_fixed_model(requests, provider, model)
        else:
            return await self._process_concurrent_fixed_model(requests, provider, model)
    
    async def _process_with_batch_api_fixed_model(self, requests: List[Tuple[str, str]], 
                                                provider: str, model: str) -> List[Optional[str]]:
        """Process requests using batch API with fixed model."""
        # Anthropic python SDK currently lacks a stable files/batches API.
        # Use concurrent processing for Anthropic to avoid attribute errors.
        if provider == "anthropic":
            return await self._process_anthropic_batch_fixed_model(requests, model)

        try:
            if provider == "openai":
                return await self._process_openai_batch_fixed_model(requests, model)
            else:
                # Fallback to concurrent processing
                return await self._process_concurrent_fixed_model(requests, provider, model)
        except Exception as e:
            print(f"Batch API failed: {e}, falling back to concurrent processing")
            return await self._process_concurrent_fixed_model(requests, provider, model)

    async def _process_anthropic_batch_fixed_model(self, requests: List[Tuple[str, str]], model: str) -> List[Optional[str]]:
        """Process batch using Anthropic Message Batches API via SDK."""
        try:
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request as AnthRequest
        except Exception as e:
            print(f"ℹ️  Anthropic batch types unavailable ({e}); using concurrent processing")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)

        client = self.client_manager.get_client("anthropic")
        if not client:
            print("❌ Anthropic client not available; using concurrent processing")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)

        # Build SDK requests
        anth_requests = []
        for i, (system, prompt) in enumerate(requests):
            params = MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            anth_requests.append(AnthRequest(custom_id=f"request-{i}", params=params))

        try:
            batch = await asyncio.to_thread(client.messages.batches.create, requests=anth_requests)
            batch_id = getattr(batch, 'id', None) or (isinstance(batch, dict) and batch.get('id'))
            print(f"📝 Anthropic batch created: {batch_id} • model={model} • {len(requests)} requests")

            if not getattr(self.config, 'auto_retrieve_batches', True):
                return [f"BATCH_SUBMITTED:{batch_id}"] * len(requests)

            # Poll until ended
            poll_interval = 10
            waited = 0
            max_wait_seconds = 60 * 60 * 24
            def get_status(obj):
                return getattr(obj, 'processing_status', None) or (isinstance(obj, dict) and obj.get('processing_status')) or 'unknown'

            status = get_status(batch)
            while status not in ('ended',) and waited < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                batch = await asyncio.to_thread(client.messages.batches.retrieve, batch_id)
                status = get_status(batch)
                print(f"⏳ Anthropic batch {batch_id} status: {status} (waited {waited}s)")

            if status != 'ended':
                print(f"❌ Anthropic batch {batch_id} did not end (status={status})")
                return [None] * len(requests)

            # Stream results
            outputs: Dict[int, Optional[str]] = {i: None for i in range(len(requests))}

            def consume_results():
                try:
                    for result in client.messages.batches.results(batch_id):
                        # result.custom_id, result.result.type, and message content
                        cid = getattr(result, 'custom_id', None) or (hasattr(result, 'get') and result.get('custom_id'))
                        res = getattr(result, 'result', None) or (hasattr(result, 'get') and result.get('result'))
                        if not cid or not str(cid).startswith('request-') or res is None:
                            continue
                        try:
                            idx = int(str(cid).split('request-')[1])
                        except Exception:
                            continue
                        rtype = getattr(res, 'type', None) or (isinstance(res, dict) and res.get('type'))
                        if rtype != 'succeeded':
                            continue
                        # message with content list
                        msg = getattr(res, 'message', None) or (isinstance(res, dict) and res.get('message'))
                        content = getattr(msg, 'content', None) or (isinstance(msg, dict) and msg.get('content'))
                        text = None
                        if isinstance(content, list) and content:
                            first = content[0]
                            text = getattr(first, 'text', None) or (isinstance(first, dict) and first.get('text'))
                        outputs[idx] = text
                except Exception as e:
                    print(f"⚠️  Failed to consume Anthropic batch results: {e}")
                return outputs

            outputs = await asyncio.to_thread(consume_results)
            return [outputs.get(i) for i in range(len(requests))]
        except Exception as e:
            print(f"❌ Anthropic batch processing error: {e}")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)
    
    async def _process_openai_batch_fixed_model(self, requests: List[Tuple[str, str]], model: str) -> List[Optional[str]]:
        """Process batch using OpenAI Batch API with fixed model."""
        client = self.client_manager.get_client("openai")
        if not client:
            raise ValueError("OpenAI client not available")
        
        print(f"🔍 DEBUG: _process_openai_batch_fixed_model received {len(requests)} requests")
        
        temperature = self.client_manager.get_model_temperature(model)
        
        # Create batch file with structured output schema (unique timestamp + random)
        import random
        unique_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        batch_file_path = f"/tmp/openai_batch_{unique_id}.jsonl"
        batch_requests = []

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "dlp_example",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "channel": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "recipients": {"type": "array", "items": {"type": "string"}},
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
                        "links": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["channel", "subject", "body", "recipients", "attachments", "links"]
                }
            }
        }

        for i, (system, prompt) in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system + "\nRespond ONLY with valid JSON matching the schema. Do not include 'Turn N' or meta-instructions."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "response_format": response_format
                }
            }
            batch_requests.append(batch_request)
        
        print(f"🔍 DEBUG: Created {len(batch_requests)} batch requests from {len(requests)} input requests")
        
        # Write batch request file (JSONL)
        with open(batch_file_path, 'w') as f:
            for req in batch_requests:
                f.write(json.dumps(req) + '\n')
                
        print(f"🔍 DEBUG: Wrote batch file to {batch_file_path}")
        
        # Check actual file size
        import os
        file_size = os.path.getsize(batch_file_path)
        with open(batch_file_path, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"🔍 DEBUG: Batch file has {line_count} lines, {file_size} bytes")

        # Upload file and create batch
        try:
            # Upload + create with retries
            retries = 3
            delay = 2
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    uploaded = await asyncio.to_thread(
                        client.files.create,
                        file=open(batch_file_path, 'rb'),
                        purpose='batch'
                    )
                    batch = await asyncio.to_thread(
                        client.batches.create,
                        input_file_id=uploaded.id,
                        endpoint='/v1/chat/completions',
                        completion_window='24h',
                        metadata={
                            'source': 'hrm_dlp_agentic',
                            'model': model,
                            'count': str(len(requests))
                        }
                    )
                    break
                except Exception as e:
                    last_err = e
                    print(f"OpenAI batch create attempt {attempt}/{retries} failed: {e}")
                    if attempt < retries:
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        raise last_err

            print(f"📝 OpenAI batch created: {batch.id} • model={model} • {len(requests)} requests")

            # Persist a simple record for operator visibility
            try:
                record = {
                    'batch_id': batch.id,
                    'model': model,
                    'request_count': len(requests),
                    'temperature': temperature,
                    'file_path': batch_file_path,
                }
                with open('openai-batches.json', 'a') as rf:
                    rf.write(json.dumps(record) + '\n')
            except Exception:
                pass

            # If auto-retrieval disabled, return placeholders now
            if not getattr(self.config, 'auto_retrieve_batches', True):
                return [f"BATCH_SUBMITTED:{batch.id}"] * len(requests)

            # Poll for completion (pause-aware)
            poll_interval = 10
            max_wait_seconds = 60 * 60 * 24  # 24 hours maximum timeout
            waited = 0
            status = getattr(batch, 'status', 'created')
            while status in ('validating', 'in_progress', 'finalizing', 'created') and waited < max_wait_seconds:
                # Respect pause/resume if configured
                pause_event = getattr(self.config, 'recovery_pause_event', None)
                if pause_event is not None and not pause_event.is_set():
                    print(f"⏸️  OpenAI batch {batch.id} polling paused. Waiting to resume...")
                    await pause_event.wait()
                    print(f"▶️  Resuming OpenAI batch {batch.id} polling")
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                batch = await asyncio.to_thread(client.batches.retrieve, batch.id)
                status = getattr(batch, 'status', 'unknown')
                print(f"⏳ OpenAI batch {batch.id} status: {status} (waited {waited}s)")

            if status != 'completed':
                print(f"❌ OpenAI batch {batch.id} did not complete (status={status})")
                return [None] * len(requests)

            # Fetch output file content (retry if id not immediately available)
            output_file_id = getattr(batch, 'output_file_id', None)
            if not output_file_id:
                # Some latency before output_file_id becomes available; retry briefly
                for _ in range(12):  # up to ~60s
                    await asyncio.sleep(5)
                    batch = await asyncio.to_thread(client.batches.retrieve, batch.id)
                    output_file_id = getattr(batch, 'output_file_id', None)
                    if output_file_id:
                        break
                if not output_file_id:
                    err_id = getattr(batch, 'error_file_id', None)
                    if err_id:
                        print(f"❌ OpenAI batch {batch.id} completed but no output_file_id; error_file_id={err_id}")
                    else:
                        print(f"❌ OpenAI batch {batch.id} completed without output_file_id")
                    return [None] * len(requests)

            try:
                file_content = await asyncio.to_thread(client.files.content, output_file_id)
                # Some clients return a Response-like object; try to get text
                content_text = getattr(file_content, 'text', None)
                if content_text is None:
                    # Fallback: read bytes-like
                    try:
                        content_text = file_content.read().decode('utf-8')
                    except Exception:
                        content_text = str(file_content)
            except Exception as e:
                print(f"❌ Failed to retrieve output for batch {batch.id}: {e}")
                return [None] * len(requests)

            # Parse JSONL output mapping by custom_id
            outputs: Dict[int, Optional[str]] = {i: None for i in range(len(requests))}
            success_count = 0
            error_count = 0
            
            try:
                for line in content_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    custom_id = rec.get('custom_id') or (rec.get('id') and rec['id'])
                    if not custom_id or not str(custom_id).startswith('request-'):
                        continue
                    try:
                        idx = int(str(custom_id).split('request-')[1])
                    except Exception:
                        continue
                    # Success path
                    if 'response' in rec and rec['response'].get('status_code') == 200:
                        body = rec['response'].get('body') or {}
                        # OpenAI chat completion body
                        try:
                            content = body['choices'][0]['message']['content']
                            outputs[idx] = content
                            success_count += 1
                        except Exception:
                            # Fallback: store the raw body
                            outputs[idx] = json.dumps(body)
                            success_count += 1
                    else:
                        # Error path; log the error and leave None
                        error_info = rec.get('response', {})
                        status_code = error_info.get('status_code', 'unknown')
                        error_body = error_info.get('body', {})
                        error_msg = error_body.get('error', {}).get('message', 'Unknown error')
                        
                        # Log first few errors to understand the pattern
                        if error_count < 5:  # Only log first 5 errors to avoid spam
                            print(f"❌ Request {idx} failed: status={status_code}, error={error_msg}")
                        
                        outputs[idx] = None
                        error_count += 1
                        
                # Summary of batch results
                print(f"📊 Batch {batch.id} results: ✅ {success_count} success, ❌ {error_count} errors")
                
            except Exception as e:
                print(f"⚠️  Failed to parse batch output file for {batch.id}: {e}")

            # Return outputs ordered by original request index
            return [outputs[i] for i in range(len(requests))]
        except Exception as e:
            print(f"❌ Failed to create OpenAI batch: {e}")
            # Fall back to concurrent processing with the same model
            return await self._process_concurrent_fixed_model(requests, 'openai', model)
    
    async def _process_anthropic_batch_fixed_model(self, requests: List[Tuple[str, str]], model: str) -> List[Optional[str]]:
        """Process batch using Anthropic Messages Batches API (SDK, no files)."""
        try:
            import anthropic  # noqa: F401
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request as AnthRequest
        except Exception as e:
            print(f"ℹ️  Anthropic batch types unavailable ({e}); please upgrade anthropic SDK. Using concurrent processing.")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)

        client = self.client_manager.get_client("anthropic")
        if not client:
            print("❌ Anthropic client not available; using concurrent processing")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)

        # Build SDK requests (embed system into user content for compatibility)
        anth_requests = []
        for i, (system, prompt) in enumerate(requests):
            params = MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": f"{system}\n\n{prompt}"}],
            )
            anth_requests.append(AnthRequest(custom_id=f"request-{i}", params=params))

        try:
            batch = await asyncio.to_thread(client.messages.batches.create, requests=anth_requests)
            batch_id = getattr(batch, 'id', None) or (isinstance(batch, dict) and batch.get('id'))
            print(f"📝 Anthropic batch created: {batch_id} • model={model} • {len(requests)} requests")

            if not getattr(self.config, 'auto_retrieve_batches', True):
                return [f"BATCH_SUBMITTED:{batch_id}"] * len(requests)

            # Poll until ended
            def get_status(obj):
                return getattr(obj, 'processing_status', None) or (isinstance(obj, dict) and obj.get('processing_status')) or 'unknown'

            poll_interval = 10
            waited = 0
            max_wait_seconds = 60 * 60 * 24
            status = get_status(batch)
            while status not in ('ended',) and waited < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                batch = await asyncio.to_thread(client.messages.batches.retrieve, batch_id)
                status = get_status(batch)
                print(f"⏳ Anthropic batch {batch_id} status: {status} (waited {waited}s)")

            if status != 'ended':
                print(f"❌ Anthropic batch {batch_id} did not end (status={status})")
                return [None] * len(requests)

            # Stream results and map by custom_id
            outputs: Dict[int, Optional[str]] = {i: None for i in range(len(requests))}

            def consume_results():
                try:
                    for result in client.messages.batches.results(batch_id):
                        cid = getattr(result, 'custom_id', None) or (isinstance(result, dict) and result.get('custom_id'))
                        res = getattr(result, 'result', None) or (isinstance(result, dict) and result.get('result'))
                        if not cid or not str(cid).startswith('request-') or res is None:
                            continue
                        try:
                            idx = int(str(cid).split('request-')[1])
                        except Exception:
                            continue
                        rtype = getattr(res, 'type', None) or (isinstance(res, dict) and res.get('type'))
                        if rtype != 'succeeded':
                            continue
                        msg = getattr(res, 'message', None) or (isinstance(res, dict) and res.get('message'))
                        content = getattr(msg, 'content', None) or (isinstance(msg, dict) and msg.get('content'))
                        text = None
                        if isinstance(content, list) and content:
                            first = content[0]
                            text = getattr(first, 'text', None) or (isinstance(first, dict) and first.get('text'))
                        outputs[idx] = text
                except Exception as e:
                    print(f"⚠️  Failed to consume Anthropic batch results: {e}")
                return outputs

            outputs = await asyncio.to_thread(consume_results)
            return [outputs.get(i) for i in range(len(requests))]
        except Exception as e:
            print(f"❌ Anthropic batch processing error: {e}")
            return await self._process_concurrent_fixed_model(requests, 'anthropic', model)
    
    async def _process_concurrent_fixed_model(self, requests: List[Tuple[str, str]], 
                                            provider: str, model: str) -> List[Optional[str]]:
        """Process requests concurrently with fixed model."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        client = self.client_manager.get_client(provider)
        
        if not client:
            print(f"❌ Client not available for {provider}")
            return [None] * len(requests)
        
        temperature = self.client_manager.get_model_temperature(model)
        
        async def process_single_request(system: str, prompt: str) -> Optional[str]:
            async with semaphore:
                attempts = getattr(self.config, 'max_retries', 3)
                delay = 0.2
                for attempt in range(1, attempts + 1):
                    try:
                        # Add small delay to avoid rate limits
                        await asyncio.sleep(0.1)
                        if provider == "anthropic":
                            response = await asyncio.to_thread(
                                client.messages.create,
                                model=model,
                                max_tokens=4096,
                                temperature=temperature,
                                system=system + "\nRespond ONLY with valid JSON fields: subject, body, recipients.",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            return response.content[0].text
                        else:
                            # OpenAI and compatible providers
                            kwargs = {
                                'model': model,
                                'messages': [
                                    {"role": "system", "content": system + "\nRespond ONLY with valid JSON matching the schema. Do not include 'Turn N' or meta-instructions."},
                                    {"role": "user", "content": prompt}
                                ],
                                'temperature': temperature
                            }
                            if provider == 'openai':
                                kwargs['response_format'] = {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "dlp_example",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "channel": {"type": "string"},
                                                "subject": {"type": "string"},
                                                "body": {"type": "string"},
                                                "recipients": {"type": "array", "items": {"type": "string"}},
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
                                                "links": {"type": "array", "items": {"type": "string"}},
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
                                            "required": ["channel", "subject", "body", "recipients", "attachments", "links", "user", "context_summary", "thread"]
                                        }
                                    }
                                }
                            response = await asyncio.to_thread(
                                client.chat.completions.create,
                                **kwargs
                            )
                            return response.choices[0].message.content
                    except Exception as e:
                        print(f"Request attempt {attempt}/{attempts} failed with {provider}/{model}: {e}")
                        if attempt < attempts:
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            return None
        
        # Process all requests concurrently with the same model
        print(f"🔄 Processing {len(requests)} requests concurrently with {provider}/{model}")
        tasks = [process_single_request(system, prompt) for system, prompt in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        processed_results = [result if not isinstance(result, Exception) else None for result in results]
        
        # Count successful requests
        successful = sum(1 for r in processed_results if r is not None)
        print(f"✅ Concurrent processing complete: {successful}/{len(requests)} successful with {provider}/{model}")
        
        return processed_results
    
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
    
    def get_model_info_for_batch(self, requests: List[Any]) -> Tuple[str, str, Dict]:
        """Get recommended model info for a batch of requests."""
        # Analyze request characteristics
        total_chars = sum(len(str(req)) for req in requests)
        avg_chars = total_chars / len(requests) if requests else 0
        
        # Choose model based on complexity
        if avg_chars > 2000:  # Complex requests
            preferred_provider = "anthropic"
            reason = "complex_requests"
        elif len(requests) > 1000:  # Large batch
            preferred_provider = "openai" 
            reason = "large_batch"
        else:
            preferred_provider = "anthropic"  # Default
            reason = "default"
        
        provider, model = self.client_manager.choose_model(preferred_provider)
        
        return provider, model, {
            "reason": reason,
            "total_chars": total_chars,
            "avg_chars": avg_chars,
            "request_count": len(requests)
        }
