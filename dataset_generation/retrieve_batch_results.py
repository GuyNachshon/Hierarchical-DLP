#!/usr/bin/env python3
"""
Retrieve and process completed batch results from Anthropic API.

This script fetches results from completed batches and integrates them
into the checkpoint system for immediate use.
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append('.')

from batch_tracker import BatchTracker
from agentic_data_generator import AgenticDataGenerator, AgenticConfig, GeneratedExample


def retrieve_anthropic_batch_results(client, batch_id: str):
    """Retrieve results from a completed Anthropic batch"""
    try:
        print(f"📡 Fetching results for batch {batch_id[:8]}...")

        # Get batch details
        batch = client.messages.batches.retrieve(batch_id)
        print(f"📊 Batch status: {batch.processing_status}")

        if batch.processing_status != "ended":
            print(f"⚠️ Batch not yet completed: {batch.processing_status}")
            return []

        print(f"📊 Request counts: {batch.request_counts}")

        # Get results using the results URL
        results = []
        if hasattr(batch, 'results_url') and batch.results_url:
            print(f"📡 Downloading results from: {batch.results_url}")

            # Use Anthropic client to download results (handles auth)
            import requests
            headers = {
                'x-api-key': client.api_key,
                'anthropic-version': '2023-06-01'
            }

            response = requests.get(batch.results_url, headers=headers)
            if response.status_code == 200:
                print(f"✅ Downloaded {len(response.text)} bytes of results")

                # Process each line of JSONL results
                for line_num, line in enumerate(response.text.strip().split('\n')):
                    if not line:
                        continue

                    try:
                        result_data = json.loads(line)
                        print(f"📝 Processing result {line_num + 1}...")

                        if 'result' in result_data:
                            result = result_data['result']
                            if result['type'] == 'succeeded' and 'message' in result:
                                message = result['message']
                                if 'content' in message and len(message['content']) > 0:
                                    content = message['content'][0]['text']
                                    results.append(content)
                                    print(f"✅ Extracted content ({len(content)} chars)")
                                else:
                                    results.append(None)
                                    print("⚠️ No content in message")
                            elif result['type'] == 'errored':
                                print(f"❌ Request failed: {result.get('error', 'Unknown error')}")
                                results.append(None)
                            else:
                                results.append(None)
                                print(f"⚠️ Unexpected result type: {result.get('type', 'unknown')}")
                        else:
                            results.append(None)
                            print("⚠️ No result in response")

                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error on line {line_num + 1}: {e}")
                        results.append(None)
            else:
                print(f"❌ Failed to download results: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return []
        else:
            print("❌ No results_url available in batch")
            return []

        successful_results = [r for r in results if r is not None]
        print(f"📦 Retrieved {len(results)} total results, {len(successful_results)} successful")
        return results

    except Exception as e:
        print(f"❌ Error retrieving batch results: {e}")
        import traceback
        traceback.print_exc()
        return []


def integrate_batch_results(generator: AgenticDataGenerator, batch_id: str, results: list, provider: str, model: str):
    """Integrate batch results into the checkpoint system"""
    try:
        print(f"🔗 Integrating {len(results)} batch results...")

        processed_count = 0
        failed_count = 0

        for i, raw_result in enumerate(results):
            if raw_result is None:
                failed_count += 1
                continue

            try:
                # Parse the result
                example = generator.batch_processor._basic_parse_async_result(raw_result, None)
                if not example:
                    print(f"⚠️ Failed to parse result {i + 1}/{len(results)}")
                    failed_count += 1
                    continue

                # Calculate quality score
                quality_score = generator.manager.evaluate_example_quality(example)
                if quality_score < 0.3:
                    print(f"⚠️ Low quality example rejected (score: {quality_score:.2f})")
                    failed_count += 1
                    continue

                # Convert to dictionary format
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
                        "provider": provider,
                        "model": model,
                        "recovered_at": time.time()
                    }
                }

                # Save to checkpoint
                if generator.state_manager:
                    generator.state_manager.append_completed_example(example_dict)
                    processed_count += 1
                    print(f"💾 Saved example {processed_count}/{len(results)}")
                else:
                    print("⚠️ No state manager available")

            except Exception as e:
                print(f"⚠️ Error processing result {i + 1}: {e}")
                failed_count += 1

        print(f"✅ Batch integration complete: {processed_count} saved, {failed_count} failed")
        return processed_count

    except Exception as e:
        print(f"❌ Error integrating batch results: {e}")
        return 0


def main():
    print("🚀 Starting batch result retrieval and recovery...")

    # Initialize components
    config = AgenticConfig()
    config.checkpoint_dir = 'data/dlp_agentic/.checkpoints'

    generator = AgenticDataGenerator(config)

    # Get Anthropic client
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    except Exception as e:
        print(f"❌ Failed to initialize Anthropic client: {e}")
        return

    # Process the two completed batches
    batch_ids = [
        'msgbatch_01EzMmiVPZBS6JGgx1cGXxHK',
        'msgbatch_01HftmyQwfwDZbusCj2vvmYt'
    ]

    total_processed = 0

    for batch_id in batch_ids:
        print(f"\\n🔄 Processing batch {batch_id[:8]}...")

        # Retrieve results
        results = retrieve_anthropic_batch_results(client, batch_id)

        if results:
            # Integrate results
            processed = integrate_batch_results(
                generator, batch_id, results,
                'anthropic', 'claude-3-haiku-20240307'
            )
            total_processed += processed
        else:
            print(f"⚠️ No results retrieved for batch {batch_id[:8]}")

    print(f"\\n🎉 Recovery completed: {total_processed} examples processed and saved")
    print(f"📁 Checkpoint file updated: {config.checkpoint_dir}/completed_examples.jsonl")


if __name__ == "__main__":
    main()
