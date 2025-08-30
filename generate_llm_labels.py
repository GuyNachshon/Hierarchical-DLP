#!/usr/bin/env python3
"""
Generate LLM-Based DLP Labels

Uses OpenAI/Anthropic models to generate high-quality, contextually-aware 
DLP labels for the entire training dataset.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse
from datetime import datetime

from llm_api_integration import LLMAPIClient, LLMLabelingConfig, LabelingResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMDatasetLabeler:
    """Main class for labeling entire datasets with LLM."""
    
    def __init__(self, config: LLMLabelingConfig):
        self.config = config
        self.client = LLMAPIClient(config)
        self.results_log = []
        
    def load_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        examples = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        return examples
    
    def save_labeled_dataset(self, examples: List[Dict[str, Any]], 
                           results: List[LabelingResult], 
                           output_path: Path):
        """Save dataset with LLM-generated labels."""
        
        successful_count = 0
        failed_count = 0
        
        with open(output_path, 'w') as f:
            for example, result in zip(examples, results):
                if result.success and result.labels:
                    # Add LLM labels to example
                    example_copy = example.copy()
                    example_copy['llm_labels'] = result.labels
                    
                    # Add reasoning if available
                    if result.reasoning:
                        example_copy['llm_reasoning'] = result.reasoning
                    
                    # Add metadata
                    example_copy['llm_metadata'] = {
                        'provider': self.config.provider,
                        'model': self.config.model,
                        'cost': result.cost_estimate,
                        'response_time': result.response_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    f.write(json.dumps(example_copy) + '\n')
                    successful_count += 1
                else:
                    # Keep original example but log failure
                    f.write(json.dumps(example) + '\n')
                    failed_count += 1
                    logger.warning(f"Failed to label example: {result.error}")
        
        logger.info(f"Saved {successful_count} successfully labeled examples to {output_path}")
        if failed_count > 0:
            logger.warning(f"{failed_count} examples failed to be labeled")
    
    def progress_callback(self, current: int, total: int, result: LabelingResult):
        """Progress callback for batch processing."""
        if current % 10 == 0 or current == total:
            stats = self.client.get_statistics()
            print(f"Progress: {current}/{total} ({current/total*100:.1f}%) | "
                  f"Success: {stats['success_rate']:.1f}% | "
                  f"Cost: ${stats['total_cost']:.3f}")
            
        # Log result for analysis
        self.results_log.append({
            'index': current - 1,
            'success': result.success,
            'cost': result.cost_estimate,
            'response_time': result.response_time,
            'error': result.error if not result.success else None
        })
    
    async def label_dataset(self, input_path: Path, output_path: Path, 
                           max_examples: Optional[int] = None) -> Dict[str, Any]:
        """Label entire dataset with LLM."""
        
        print(f"🤖 LLM Dataset Labeling")
        print(f"   Provider: {self.config.provider}")
        print(f"   Model: {self.config.model}")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        
        # Load dataset
        examples = self.load_dataset(input_path)
        
        if max_examples:
            examples = examples[:max_examples]
            print(f"   Limiting to first {max_examples} examples")
        
        # Estimate cost
        estimated_cost = len(examples) * self._estimate_cost_per_example()
        print(f"   Estimated cost: ${estimated_cost:.2f}")
        
        # Confirm before proceeding
        if estimated_cost > 10:  # Alert for expensive runs
            response = input(f"\n⚠️  Estimated cost is ${estimated_cost:.2f}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted by user")
                return {}
        
        # Process in batches
        print(f"\n🔄 Processing {len(examples)} examples...")
        
        batch_size = self.config.batch_size
        all_results = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(examples) + batch_size - 1) // batch_size
            
            print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} examples)")
            
            batch_results = await self.client.label_batch(
                batch, 
                progress_callback=self.progress_callback
            )
            
            all_results.extend(batch_results)
        
        # Save results
        print(f"\n💾 Saving labeled dataset...")
        self.save_labeled_dataset(examples, all_results, output_path)
        
        # Generate summary
        stats = self.client.get_statistics()
        summary = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'total_examples': len(examples),
            'processing_stats': stats,
            'config': {
                'provider': self.config.provider,
                'model': self.config.model,
                'temperature': self.config.temperature
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _estimate_cost_per_example(self) -> float:
        """Estimate cost per example based on model."""
        # Rough estimates based on typical email lengths
        base_costs = {
            "gpt-4o": 0.05,
            "gpt-4o-mini": 0.005,
            "gpt-4-turbo": 0.08,
            "claude-3-5-sonnet-20241022": 0.04,
            "claude-3-haiku-20240307": 0.003,
        }
        
        return base_costs.get(self.config.model, 0.02)  # Default estimate

def analyze_existing_labels(file_path: Path):
    """Analyze existing rule-based labels for comparison."""
    print(f"\n📊 Analyzing existing labels in {file_path.name}...")
    
    if not file_path.exists():
        print(f"   ❌ File not found: {file_path}")
        return
    
    labels_data = {'sensitivity': [], 'exposure': [], 'context': [], 'obfuscation': []}
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                example = json.loads(line)
                if 'labels' in example:
                    for label_type, values_list in labels_data.items():
                        if label_type in example['labels']:
                            values_list.append(example['labels'][label_type])
            except:
                continue
    
    if any(labels_data.values()):
        print("   📈 Existing Rule-Based Labels:")
        for label_type, values in labels_data.items():
            if values:
                import numpy as np
                values = np.array(values)
                print(f"      {label_type.capitalize():<12}: mean={values.mean():.3f}, std={values.std():.3f}, range={values.max()-values.min():.3f}")
    else:
        print("   ℹ️  No existing labels found")

async def main():
    """Main function for LLM labeling."""
    parser = argparse.ArgumentParser(description="Generate LLM-based DLP labels")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                       help="LLM provider to use")
    parser.add_argument("--model", type=str, 
                       help="Model to use (e.g., gpt-4o-mini, claude-3-haiku-20240307)")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file")
    parser.add_argument("--output", type=str,
                       help="Output JSONL file (default: input_llm_labeled.jsonl)")
    parser.add_argument("--max_examples", type=int,
                       help="Maximum number of examples to process (for testing)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="LLM temperature (lower = more consistent)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--rate_limit", type=float, default=1.0,
                       help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_llm_labeled.jsonl"
    
    # Set default models
    if not args.model:
        if args.provider == "openai":
            args.model = "gpt-4o-mini"  # Cheaper option
        else:
            args.model = "claude-3-haiku-20240307"  # Cheaper option
    
    # Create configuration
    config = LLMLabelingConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        rate_limit_delay=args.rate_limit
    )
    
    # Analyze existing labels
    analyze_existing_labels(input_path)
    
    # Create labeler and process
    labeler = LLMDatasetLabeler(config)
    
    try:
        summary = await labeler.label_dataset(input_path, output_path, args.max_examples)
        
        # Print summary
        print(f"\n✅ LLM Labeling Complete!")
        print(f"   📁 Output: {output_path}")
        print(f"   📊 Processed: {summary['total_examples']} examples")
        print(f"   💰 Total cost: ${summary['processing_stats']['total_cost']:.3f}")
        print(f"   ⚡ Success rate: {summary['processing_stats']['success_rate']:.1f}%")
        
        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📝 Summary saved: {summary_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 LLM-Based DLP Label Generation")
    print("=" * 60)
    print("This script generates high-quality, contextually-aware DLP labels")
    print("using OpenAI or Anthropic language models.")
    print()
    print("Setup required:")
    print("   1. Install dependencies: pip install openai anthropic")
    print("   2. Set API key: export OPENAI_API_KEY='your-key' or ANTHROPIC_API_KEY='your-key'")
    print("   3. Run: python generate_llm_labels.py --input data/file.jsonl")
    print()
    
    # Check if we're being run directly
    import sys
    if len(sys.argv) > 1:
        asyncio.run(main())
    else:
        print("Usage examples:")
        print("  python generate_llm_labels.py --input data/hrm_dlp_final/train.jsonl --max_examples 10")
        print("  python generate_llm_labels.py --input data/hrm_dlp_final/train.jsonl --provider anthropic")
        print("  python generate_llm_labels.py --input data/hrm_dlp_final/train.jsonl --model gpt-4o")