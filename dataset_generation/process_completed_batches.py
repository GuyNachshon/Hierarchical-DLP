#!/usr/bin/env python3
"""
Process completed batch results and integrate them into checkpoint system.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, List

load_dotenv()
sys.path.append('.')

from agentic_data_generator import GeneratedExample

def parse_llm_result(raw_text: str) -> Optional[GeneratedExample]:
    """Parse LLM result text into GeneratedExample"""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', raw_text, re.DOTALL)
        
        if json_match:
            parsed = json.loads(json_match.group(1))
            
            return GeneratedExample(
                channel=parsed.get("channel", "email"),
                user=parsed.get("user", {"role": "batch_recovered"}),
                recipients=parsed.get("recipients", []),
                subject=parsed.get("subject", ""),
                body=parsed.get("body", raw_text[:500] + "..." if len(raw_text) > 500 else raw_text),
                attachments=parsed.get("attachments", []),
                links=parsed.get("links", []),
                thread=parsed.get("thread"),
                labels=parsed.get("labels"),
                spans=parsed.get("spans"),
                meta={
                    "agent_type": "batch_recovered",
                    "source": "batch_recovery",
                    "original_length": len(raw_text)
                }
            )
        else:
            # Fallback: create example from raw text
            return GeneratedExample(
                channel="email",
                user={"role": "batch_recovered_fallback"},
                recipients=["recipient@example.com"],
                subject="Recovered Batch Result",
                body=raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
                attachments=[],
                links=[],
                thread=None,
                labels=None,
                spans=None,
                meta={
                    "agent_type": "batch_recovered_fallback",
                    "source": "batch_recovery_fallback",
                    "original_length": len(raw_text),
                    "parsing_failed": True
                }
            )
    except Exception as e:
        print(f"⚠️ Error parsing result: {e}")
        return None

def calculate_quality_score(example: GeneratedExample) -> float:
    """Calculate quality score for an example"""
    score = 0.5  # Base score
    
    # Check required fields
    if example.body and len(example.body) > 20:
        score += 0.2
    if example.subject and len(example.subject) > 5:
        score += 0.1
    if example.recipients:
        score += 0.1
    if example.spans:
        score += 0.1
        
    return min(1.0, score)

def save_example_to_checkpoint(example_dict: Dict, checkpoint_file: Path):
    """Save example to checkpoint file"""
    try:
        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example_dict) + '\n')
        return True
    except Exception as e:
        print(f"❌ Failed to save example: {e}")
        return False

def retrieve_and_process_anthropic_batch(batch_id: str, checkpoint_dir: str):
    """Retrieve and process a single Anthropic batch"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        print(f"\\n📡 Processing batch {batch_id[:8]}...")
        
        # Get batch details
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            print(f"⚠️ Batch not completed: {batch.processing_status}")
            return 0
        
        # Download results
        import requests
        headers = {
            'x-api-key': client.api_key,
            'anthropic-version': '2023-06-01'
        }
        
        response = requests.get(batch.results_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to download results: HTTP {response.status_code}")
            return 0
        
        print(f"✅ Downloaded {len(response.text)} bytes of results")
        
        # Save to temporary file for proper JSONL parsing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write(response.text)
            temp_file_path = temp_file.name
        
        # Process results
        checkpoint_file = Path(checkpoint_dir) / "completed_examples.jsonl"
        processed_count = 0
        failed_count = 0
        
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        result_data = json.loads(line)
                        
                        if 'result' in result_data:
                            result = result_data['result']
                            if result['type'] == 'succeeded' and 'message' in result:
                                message = result['message']
                                if 'content' in message and len(message['content']) > 0:
                                    raw_text = message['content'][0]['text']
                                    
                                    # Parse into GeneratedExample
                                    example = parse_llm_result(raw_text)
                                    if not example:
                                        failed_count += 1
                                        continue
                                    
                                    # Calculate quality score
                                    quality_score = calculate_quality_score(example)
                                    if quality_score < 0.3:
                                        print(f"⚠️ Low quality example rejected (score: {quality_score:.2f})")
                                        failed_count += 1
                                        continue
                                    
                                    # Convert to dict and save
                                    example_dict = {
                                        "channel": example.channel,
                                        "user": example.user,
                                        "recipients": example.recipients,
                                        "subject": example.subject,
                                        "body": example.body,
                                        "attachments": example.attachments,
                                        "links": example.links,
                                        "thread": example.thread,
                                        "labels": example.labels,
                                        "spans": example.spans,
                                        "meta": {
                                            **(example.meta or {}),
                                            "batch_id": batch_id[:8],
                                            "quality_score": quality_score,
                                            "recovered_batch": True,
                                            "provider": "anthropic",
                                            "model": "claude-3-haiku-20240307",
                                            "recovered_at": time.time()
                                        }
                                    }
                                    
                                    if save_example_to_checkpoint(example_dict, checkpoint_file):
                                        processed_count += 1
                                        if processed_count % 10 == 0:
                                            print(f"💾 Saved {processed_count} examples...")
                                    else:
                                        failed_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        print(f"⚠️ Error processing line {line_num + 1}: {e}")
                        failed_count += 1
        
        finally:
            # Clean up temp file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        print(f"✅ Batch {batch_id[:8]} complete: {processed_count} saved, {failed_count} failed")
        return processed_count
        
    except Exception as e:
        print(f"❌ Error processing batch {batch_id[:8]}: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    print("🚀 Processing completed Anthropic batches...")
    
    checkpoint_dir = 'data/dlp_agentic/.checkpoints'
    
    # Process the two completed batches
    batch_ids = [
        'msgbatch_01EzMmiVPZBS6JGgx1cGXxHK',
        'msgbatch_01HftmyQwfwDZbusCj2vvmYt'
    ]
    
    total_processed = 0
    
    for batch_id in batch_ids:
        processed = retrieve_and_process_anthropic_batch(batch_id, checkpoint_dir)
        total_processed += processed
    
    print(f"\\n🎉 Batch processing completed!")
    print(f"📊 Total examples processed: {total_processed}")
    print(f"📁 Saved to: {checkpoint_dir}/completed_examples.jsonl")
    
    # Check final count
    checkpoint_file = Path(checkpoint_dir) / "completed_examples.jsonl"
    if checkpoint_file.exists():
        line_count = sum(1 for _ in open(checkpoint_file))
        print(f"📈 Checkpoint file now contains: {line_count} examples")

if __name__ == "__main__":
    main()