#!/usr/bin/env python3
"""
Recovery script for missing train batch data from HRM-DLP generation run.
This script re-fetches batch results from OpenAI API and processes them into train_examples.jsonl.

Usage:
    python recover_train_batch.py --session-id run_20250828_014046_6ddd4ecb
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
import openai
from dataclasses import dataclass


@dataclass
class BatchInfo:
    batch_id: str
    provider: str
    model: str
    request_count: int
    split_name: str
    input_file: str


def load_batch_info(session_dir: Path) -> Dict[str, BatchInfo]:
    """Load batch information from completed_batches.json."""
    completed_batches_file = session_dir / "checkpoints" / "batches" / "completed_batches.json"
    
    if not completed_batches_file.exists():
        raise FileNotFoundError(f"No completed_batches.json found in {completed_batches_file}")
    
    with open(completed_batches_file, 'r') as f:
        batch_data = json.load(f)
    
    batches = {}
    for batch_id, info in batch_data.items():
        batches[batch_id] = BatchInfo(
            batch_id=batch_id,
            provider=info["provider"],
            model=info["model"],
            request_count=info["request_count"],
            split_name=info["split_name"],
            input_file=info["input_file"]
        )
    
    return batches


def fetch_openai_batch_results(batch_id: str) -> List[Dict[str, Any]]:
    """Fetch batch results from OpenAI API."""
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    print(f"🔄 Fetching results for batch {batch_id}...")
    
    try:
        # Get batch information
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} status is {batch.status}, not completed")
        
        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file")
        
        # Download the results
        result_file_response = client.files.content(batch.output_file_id)
        result_content = result_file_response.content.decode('utf-8')
        
        # Parse JSONL results
        results = []
        for line in result_content.strip().split('\n'):
            if line.strip():
                results.append(json.loads(line))
        
        print(f"✅ Retrieved {len(results)} results from batch {batch_id}")
        return results
        
    except Exception as e:
        print(f"❌ Error fetching batch results: {e}")
        raise


def process_batch_results_to_examples(
    results: List[Dict[str, Any]], 
    batch_info: BatchInfo,
    input_file_path: Path
) -> List[Dict[str, Any]]:
    """Process batch results into DLP examples format."""
    
    # Load the original input requests to get metadata
    input_requests = {}
    if input_file_path.exists():
        with open(input_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    req = json.loads(line)
                    input_requests[req["custom_id"]] = req
    
    examples = []
    
    for result in results:
        if result.get("error"):
            print(f"⚠️ Skipping failed request {result.get('custom_id', 'unknown')}: {result['error']}")
            continue
        
        custom_id = result["custom_id"]
        response = result["response"]
        
        if response["body"]["choices"][0]["message"]["content"]:
            try:
                # Parse the LLM response as JSON
                content = response["body"]["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif content.startswith("```"):
                    content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()
                
                example_data = json.loads(content)
                
                # Add comprehensive metadata
                example_data["_metadata"] = {
                    "split": batch_info.split_name,
                    "batch_num": 1,
                    "request_index": int(custom_id.split("_")[-1]) if "_" in custom_id else 0,
                    "agent_type": "unknown",  # We'll try to infer this
                    "provider": batch_info.provider,
                    "model": batch_info.model,
                    "batch_id": batch_info.batch_id
                }
                
                # Try to infer agent type from the example content
                if "legal" in example_data.get("subject", "").lower() or "contract" in example_data.get("body", "").lower():
                    example_data["_metadata"]["agent_type"] = "legal_agent"
                elif "finance" in example_data.get("subject", "").lower() or "payment" in example_data.get("body", "").lower():
                    example_data["_metadata"]["agent_type"] = "finance_agent"
                elif "hr" in example_data.get("subject", "").lower() or "employee" in example_data.get("body", "").lower():
                    example_data["_metadata"]["agent_type"] = "hr_agent"
                else:
                    example_data["_metadata"]["agent_type"] = "clean_business"
                
                examples.append(example_data)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse response for {custom_id}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Error processing {custom_id}: {e}")
                continue
    
    print(f"✅ Processed {len(examples)} valid examples from {len(results)} results")
    return examples


def save_train_examples(examples: List[Dict[str, Any]], session_dir: Path):
    """Save examples to train_examples.jsonl."""
    output_dir = session_dir / "split_outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "train_examples.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"💾 Saved {len(examples)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Recover missing train batch data")
    parser.add_argument("--session-id", required=True, help="Session ID to recover")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    
    args = parser.parse_args()
    
    # Locate session directory
    session_dir = Path("data/runs") / args.session_id
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        sys.exit(1)
    
    print(f"🔍 Processing session: {args.session_id}")
    print(f"📁 Session directory: {session_dir}")
    
    # Load batch information
    try:
        batches = load_batch_info(session_dir)
    except Exception as e:
        print(f"❌ Failed to load batch info: {e}")
        sys.exit(1)
    
    # Find the train batch
    train_batch = None
    for batch_id, batch_info in batches.items():
        if batch_info.split_name == "train":
            train_batch = batch_info
            break
    
    if not train_batch:
        print("❌ No train batch found in completed batches")
        sys.exit(1)
    
    print(f"🎯 Found train batch: {train_batch.batch_id}")
    print(f"   Model: {train_batch.provider}/{train_batch.model}")
    print(f"   Request count: {train_batch.request_count}")
    
    # Check if train_examples.jsonl already exists
    train_file = session_dir / "split_outputs" / "train_examples.jsonl"
    if train_file.exists():
        print(f"⚠️  train_examples.jsonl already exists ({train_file})")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
    
    if args.dry_run:
        print("🔍 DRY RUN - would fetch and process batch results")
        print(f"   Batch ID: {train_batch.batch_id}")
        print(f"   Expected examples: {train_batch.request_count}")
        print(f"   Output file: {train_file}")
        return
    
    # Fetch batch results from OpenAI
    try:
        results = fetch_openai_batch_results(train_batch.batch_id)
    except Exception as e:
        print(f"❌ Failed to fetch batch results: {e}")
        sys.exit(1)
    
    # Process results into examples
    input_file_path = Path(train_batch.input_file)
    if not input_file_path.is_absolute():
        input_file_path = session_dir.parent.parent / input_file_path
    
    try:
        examples = process_batch_results_to_examples(results, train_batch, input_file_path)
    except Exception as e:
        print(f"❌ Failed to process batch results: {e}")
        sys.exit(1)
    
    if not examples:
        print("❌ No valid examples generated from batch results")
        sys.exit(1)
    
    # Save examples
    try:
        save_train_examples(examples, session_dir)
        print(f"✅ Successfully recovered {len(examples)} train examples")
        
        # Verify the file was created correctly
        if train_file.exists():
            with open(train_file, 'r') as f:
                lines = sum(1 for line in f if line.strip())
            print(f"✅ Verified: {lines} examples written to {train_file}")
        
    except Exception as e:
        print(f"❌ Failed to save examples: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()