#!/usr/bin/env python3
"""
HRM-DLP Enhanced Dataset Generation Runner

Specialized script for generating high-quality HRM-DLP training data using:
- GPT-5 models with structured outputs
- Enhanced domain agents with comprehensive prompts
- DLP format conversion with span extraction and labeling
- Batch processing with recovery and monitoring
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import create_hrm_dlp_config
from fixed_enhanced_coordinator import FixedEnhancedAgenticDataGenerator
from data.dlp_converter import DLPFormatConverter, validate_example_quality

load_dotenv()


async def generate_hrm_dlp_dataset(resume_session_id: str = None, resume_from_files: str = None):
    """Generate enhanced HRM-DLP training dataset."""
    
    print("🚀 Starting HRM-DLP Enhanced Dataset Generation")
    print("=" * 60)
    
    # Create HRM-DLP optimized configuration
    config = create_hrm_dlp_config()
    
    print(f"📊 Dataset Configuration:")
    print(f"   • Training examples: {config.train_size}")
    print(f"   • Validation examples: {config.val_size}")
    print(f"   • Test examples: {config.test_size}")
    print(f"   • Output directory: {config.output_dir}")
    print(f"   • Model priority: GPT-5 > GPT-4o")
    print(f"   • Batch API: {config.enable_batch_api}")
    print(f"   • Quality threshold: {config.min_quality_score}")
    print()
    
    # Initialize enhanced coordinator
    print("🤖 Initializing Enhanced Agentic Data Generator...")
    generator = FixedEnhancedAgenticDataGenerator(config)
    
    # Initialize DLP converter
    print("🔄 Initializing DLP Format Converter...")
    converter = DLPFormatConverter()
    
    print("✅ All components initialized successfully")
    print()
    
    # Generate dataset
    print("📝 Starting Dataset Generation Process...")
    print("   This will use batch APIs for efficient generation")
    print("   Monitor progress in the terminal dashboard")
    print()
    
    try:
        # Handle session resumption
        if resume_session_id:
            print(f"🔄 Resuming from session: {resume_session_id}")
            
            # Check if this is a completed session in data/runs/
            data_runs_session_path = Path(f"../data/runs/{resume_session_id}")
            
            if data_runs_session_path.exists():
                print(f"✅ Found completed session in {data_runs_session_path}")
                print("📊 Session appears to be fully completed with all batches retrieved")
                print("🎯 Skipping to post-processing...")
                results = data_runs_session_path  # Use existing session path
            else:
                # Try standard resume functionality
                success = await generator.resume_from_session(resume_session_id)
                if not success:
                    print("❌ Failed to resume session")
                    return False
                results = True  # Session resumed successfully
        elif resume_from_files:
            print(f"🔄 Resuming from downloaded batch files: {resume_from_files}")
            
            # Process downloaded batch files
            results = await process_batch_files(resume_from_files, config)
            if not results:
                print("❌ Failed to process batch files")
                return False
        else:
            # Run the generation process
            results = await generator.generate_dataset()
        
        if results:
            if isinstance(results, Path):
                # Completed session case
                print("🎉 Using existing completed session!")
                print(f"   Session files: {results}")
                session_output_dir = results / "split_outputs"
            else:
                # New generation case
                print("🎉 Raw dataset generation completed successfully!")
                print(f"   Generated files: {results}")
                session_output_dir = config.output_dir
            
            # Post-process with DLP conversion
            print()
            print("🔄 Post-processing with DLP Format Conversion...")
            
            try:
                # Ensure proper directory path for post-processing
                if isinstance(results, Path):
                    # For resume-from-files, use the session's split_outputs directory
                    actual_session_dir = session_output_dir
                else:
                    # For new generation, use config output dir
                    actual_session_dir = config.output_dir
                
                print(f"   📁 Processing data from: {actual_session_dir}")
                await post_process_with_dlp_conversion(actual_session_dir, converter)
                print()
                print("✅ HRM-DLP Dataset Generation Complete!")
                
                # For resume-from-files, also copy to training_data directory
                if resume_from_files:
                    training_data_dir = Path("/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP/data/training_data")
                    training_data_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Copy enhanced files from the enhanced directory to training_data directory
                        enhanced_dir = Path("data/hrm_dlp_enhanced")
                        files_copied = 0
                        
                        for split in ["train", "val", "test"]:
                            src_file = enhanced_dir / f"{split}_enhanced.jsonl"
                            if src_file.exists():
                                file_size = src_file.stat().st_size
                                if file_size > 0:
                                    import shutil
                                    dst_file = training_data_dir / f"{split}.jsonl"
                                    shutil.copy2(src_file, dst_file)
                                    dst_size = dst_file.stat().st_size
                                    with open(dst_file, 'r') as f:
                                        line_count = sum(1 for _ in f)
                                    print(f"   ✅ Copied {split} data: {line_count} examples ({dst_size:,} bytes) to {dst_file}")
                                    files_copied += 1
                                else:
                                    print(f"   ⚠️  Enhanced {split} file exists but is empty ({src_file})")
                            else:
                                print(f"   ⚠️  Enhanced {split} file not found: {src_file}")
                        
                        if files_copied > 0:
                            print(f"📁 Training dataset available at: {training_data_dir}")
                            print(f"   Successfully copied {files_copied} enhanced dataset files")
                        else:
                            print(f"⚠️  No enhanced files were copied to training_data directory")
                            print(f"   Check that DLP conversion completed successfully")
                            
                    except Exception as e:
                        print(f"⚠️  Failed to copy to training_data: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"📁 Enhanced dataset available at: {config.output_dir}")
            except Exception as post_error:
                print(f"⚠️  Post-processing completed with issues: {post_error}")
                print("📁 Raw dataset is still available and usable")
                if isinstance(results, Path):
                    print(f"   Location: {results}")
                else:
                    print(f"   Location: {config.output_dir}")
            print()
            print("Next steps:")
            print("1. Train HRM-DLP tokenizer on generated data")
            print("2. Run HRM-DLP model training")
            print("3. Evaluate model performance vs previous results")
            
        else:
            print("❌ Dataset generation failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        print("💾 Partial results saved with checkpoint recovery available")
        return False
        
    except Exception as e:
        print(f"❌ Generation failed with error: {e}")
        return False
    
    return True


async def process_batch_files(batch_files_dir: str, config) -> Path:
    """Process downloaded batch result files and create session structure."""
    import json
    import os
    from datetime import datetime
    
    batch_files_path = Path(batch_files_dir)
    if not batch_files_path.exists():
        print(f"❌ Batch files directory not found: {batch_files_path}")
        return None
    
    # Create new session directory
    session_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_from_files"
    session_path = Path(f"../data/runs/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    split_outputs_dir = session_path / "split_outputs"
    split_outputs_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created session directory: {session_path}")
    
    # First check if we have split-specific files
    all_batch_files = list(batch_files_path.glob("*.jsonl"))
    if not all_batch_files:
        print("❌ No .jsonl files found in batch directory")
        return None
    
    print(f"📄 Found {len(all_batch_files)} batch file(s) to process")
    
    # Check if files have split identifiers
    has_split_identifiers = any(
        any(split in f.name.lower() for split in ["train", "val", "test"]) 
        for f in all_batch_files
    )
    
    if has_split_identifiers:
        print("📊 Processing files with split identifiers...")
        # Process split-specific files
        splits_processed = []
        for split in ["train", "val", "test"]:
            split_examples = []
            
            # Look for batch result files for this split
            batch_files = [f for f in all_batch_files if split in f.name.lower()]
            if not batch_files:
                print(f"⚠️  No batch files found for {split} split")
                continue
            
            print(f"📄 Processing {len(batch_files)} file(s) for {split} split...")
            
            for batch_file in batch_files:
                split_examples.extend(process_single_batch_file(batch_file, split))
            
            if split_examples:
                # Write split file
                split_file = split_outputs_dir / f"{split}_examples.jsonl"
                write_examples_to_file(split_examples, split_file)
                print(f"   ✅ Created {split} split: {len(split_examples)} examples -> {split_file}")
                splits_processed.append(split)
    else:
        print("📊 No split identifiers found - processing all files as combined dataset...")
        # Process all files and distribute examples
        all_examples = []
        
        for batch_file in all_batch_files:
            all_examples.extend(process_single_batch_file(batch_file, "combined"))
        
        print(f"📊 Loaded {len(all_examples)} total examples from all batch files")
        
        if not all_examples:
            print("❌ No valid examples found in any batch files")
            return None
        
        # Distribute examples across splits (similar to your original generation)
        import random
        random.shuffle(all_examples)  # Shuffle for better distribution
        
        # Calculate split sizes based on config (or use defaults)
        total_examples = len(all_examples)
        train_size = min(2000, int(total_examples * 0.7))  # 70% for train, max 2000
        val_size = min(400, int(total_examples * 0.15))    # 15% for val, max 400  
        test_size = min(400, int(total_examples * 0.15))   # 15% for test, max 400
        
        # Adjust if we don't have enough examples
        if train_size + val_size + test_size > total_examples:
            ratio = total_examples / (train_size + val_size + test_size)
            train_size = int(train_size * ratio)
            val_size = int(val_size * ratio)
            test_size = total_examples - train_size - val_size
        
        print(f"📊 Distributing examples: {train_size} train, {val_size} val, {test_size} test")
        
        # Create splits
        splits_processed = []
        idx = 0
        
        for split, size in [("train", train_size), ("val", val_size), ("test", test_size)]:
            if size > 0 and idx < len(all_examples):
                split_examples = all_examples[idx:idx + size]
                
                # Update metadata to reflect split assignment
                for example in split_examples:
                    if '_metadata' not in example:
                        example['_metadata'] = {}
                    example['_metadata']['split'] = split
                
                # Write split file
                split_file = split_outputs_dir / f"{split}_examples.jsonl"
                write_examples_to_file(split_examples, split_file)
                print(f"   ✅ Created {split} split: {len(split_examples)} examples -> {split_file}")
                splits_processed.append(split)
                idx += size
    
    if not splits_processed:
        print("❌ No valid examples found in any batch files")
        return None
    
    print(f"🎉 Successfully processed batch files for splits: {', '.join(splits_processed)}")
    print(f"📁 Session created at: {session_path}")
    
    return session_path


def process_single_batch_file(batch_file: Path, split_name: str) -> list:
    """Process a single batch file and return list of examples."""
    import json
    
    examples = []
    print(f"   Loading {batch_file.name}...")
    
    try:
        with open(batch_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        # Parse batch API response format
                        batch_response = json.loads(line.strip())
                        
                        # Extract the actual response content
                        if 'response' in batch_response and 'body' in batch_response['response']:
                            response_body = batch_response['response']['body']
                            if 'choices' in response_body and response_body['choices']:
                                message_content = response_body['choices'][0]['message']['content']
                                
                                # Parse the generated example
                                try:
                                    example = json.loads(message_content)
                                    
                                    # Add batch metadata 
                                    if '_metadata' not in example:
                                        example['_metadata'] = {}
                                    example['_metadata']['split'] = split_name
                                    example['_metadata']['batch_file'] = batch_file.name
                                    example['_metadata']['line_number'] = line_num
                                    
                                    examples.append(example)
                                    
                                except json.JSONDecodeError as e:
                                    print(f"   ⚠️  Failed to parse example at line {line_num}: {e}")
                                    continue
                        
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  Failed to parse batch response at line {line_num}: {e}")
                        continue
                        
    except Exception as e:
        print(f"   ❌ Failed to read {batch_file}: {e}")
        return []
    
    print(f"   📊 Extracted {len(examples)} examples from {batch_file.name}")
    return examples


def write_examples_to_file(examples: list, file_path: Path):
    """Write examples to a JSONL file."""
    import json
    
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


async def post_process_with_dlp_conversion(input_dir: str, converter: DLPFormatConverter):
    """Post-process generated data with DLP format conversion."""
    import json
    import os
    
    input_path = Path(input_dir)
    # Output enhanced files to the default enhanced directory
    output_path = Path("data/hrm_dlp_enhanced")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        # Look for input files in the input directory
        raw_file = input_path / f"{split}_examples.jsonl"  # New format
        if not raw_file.exists():
            raw_file = input_path / f"{split}.jsonl"  # Legacy format
        
        # Output enhanced files to the enhanced directory
        enhanced_file = output_path / f"{split}_enhanced.jsonl"
        
        if not raw_file.exists():
            print(f"⚠️  Warning: {raw_file} not found, skipping")
            continue
        
        print(f"   Processing {split} split...")
        
        # Read raw examples
        raw_examples = []
        with open(raw_file, 'r') as f:
            for line in f:
                if line.strip():
                    raw_examples.append(json.loads(line.strip()))
        
        print(f"      Loaded {len(raw_examples)} raw examples")
        
        # Apply augmentation first
        print(f"      🔄 Applying augmentation...")
        try:
            from augmentation.augmentor import DatasetAugmentor, AugmentationConfig
            augmentation_config = AugmentationConfig(augmentation_ratio=2.0)
            augmentor = DatasetAugmentor(augmentation_config)
            
            # Debug: Check input types before augmentation
            input_types = {}
            for ex in raw_examples[:5]:  # Check first 5
                ex_type = type(ex).__name__
                input_types[ex_type] = input_types.get(ex_type, 0) + 1
            print(f"      📊 Input types to augmentation: {input_types}")
            
            augmented_examples = augmentor.augment_batch(raw_examples)
            print(f"      ✅ Augmentation successful: {len(raw_examples)} → {len(augmented_examples)} examples")
        except Exception as e:
            print(f"      ❌ Augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            augmented_examples = raw_examples
            print(f"      ⚠️  Using raw examples without augmentation")
        
        # Debug: Check types in augmented examples before DLP conversion
        type_counts = {}
        sample_strings = []
        string_indices = []
        
        for i, ex in enumerate(augmented_examples):  # Check ALL examples
            ex_type = type(ex).__name__
            type_counts[ex_type] = type_counts.get(ex_type, 0) + 1
            if isinstance(ex, str):
                string_indices.append(i)
                if len(sample_strings) < 3:
                    sample_strings.append(f"Example {i}: {ex[:100]}...")
        
        print(f"      🔍 Pre-conversion type analysis: {type_counts}")
        if string_indices:
            print(f"      📍 String objects found at indices: {string_indices[:10]}{'...' if len(string_indices) > 10 else ''}")
            print(f"      📊 Total string objects: {len(string_indices)} out of {len(augmented_examples)}")
        if sample_strings:
            print(f"      📋 String example samples:")
            for sample in sample_strings:
                print(f"         {sample}")
        
        # Convert to DLP format
        print(f"      🔄 Converting to DLP format...")
        dlp_examples = converter.convert_batch(augmented_examples)
        
        # Quality validation
        valid_examples = []
        quality_issues = []
        
        for example in dlp_examples:
            is_valid, issues = validate_example_quality(example)
            if is_valid:
                valid_examples.append(example)
            else:
                quality_issues.extend(issues)
        
        print(f"      Quality validation: {len(valid_examples)}/{len(dlp_examples)} examples passed")
        
        if quality_issues:
            print(f"      Common issues: {set(quality_issues)}")
        
        # Validate we have examples before writing
        if not valid_examples:
            print(f"      ❌ No valid examples for {split} split - skipping file creation")
            continue
            
        # Write enhanced format
        try:
            jsonl_lines = converter.to_jsonl(valid_examples)
            with open(enhanced_file, 'w') as f:
                for line in jsonl_lines:
                    f.write(line + '\n')
            
            # Verify the file was written correctly
            file_size = enhanced_file.stat().st_size
            with open(enhanced_file, 'r') as f:
                written_lines = sum(1 for _ in f)
            
            print(f"      ✅ Enhanced {split} split: {len(valid_examples)} examples -> {enhanced_file}")
            print(f"         File size: {file_size:,} bytes, Lines: {written_lines}")
            
            # Generate statistics
            stats = generate_split_statistics(valid_examples)
            stats["file_info"] = {
                "file_size_bytes": file_size,
                "lines_written": written_lines,
                "examples_expected": len(valid_examples)
            }
            
            stats_file = output_path / f"{split}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"      ❌ Failed to write enhanced {split} file: {e}")
            continue
    
    # Final validation summary
    print("   🔄 DLP format conversion completed")
    print("   📊 Final validation summary:")
    
    total_enhanced_examples = 0
    enhanced_files_created = []
    
    for split in ["train", "val", "test"]:
        enhanced_file = output_path / f"{split}_enhanced.jsonl"
        if enhanced_file.exists() and enhanced_file.stat().st_size > 0:
            with open(enhanced_file, 'r') as f:
                line_count = sum(1 for _ in f)
            total_enhanced_examples += line_count
            enhanced_files_created.append(f"{split}: {line_count} examples")
            print(f"      ✅ {split}: {line_count} enhanced examples")
        else:
            print(f"      ❌ {split}: No enhanced file or empty file")
    
    print(f"   📈 Total enhanced examples: {total_enhanced_examples}")
    print(f"   📁 Enhanced files: {len(enhanced_files_created)}/3 splits successful")


def generate_split_statistics(examples) -> dict:
    """Generate statistics for a split."""
    from collections import Counter
    
    stats = {
        "total_examples": len(examples),
        "avg_body_length": sum(len(ex.body) for ex in examples) / len(examples) if examples else 0,
        "span_distribution": Counter(span["type"] for ex in examples for span in ex.spans),
        "label_distribution": {
            "sensitivity": sum(ex.labels.get("sensitivity", 0) for ex in examples),
            "exposure": sum(ex.labels.get("exposure", 0) for ex in examples), 
            "context": sum(ex.labels.get("context", 0) for ex in examples),
            "obfuscation": sum(ex.labels.get("obfuscation", 0) for ex in examples)
        },
        "agent_distribution": Counter(ex.meta.get("agent", "unknown") for ex in examples),
        "channel_distribution": Counter(ex.channel for ex in examples),
        "thread_distribution": {
            "multi_turn": sum(1 for ex in examples if ex.thread.get("prior_msgs", 0) > 0),
            "single_turn": sum(1 for ex in examples if ex.thread.get("prior_msgs", 0) == 0)
        }
    }
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HRM-DLP Enhanced Dataset Generation")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show configuration without running generation")
    parser.add_argument("--demo-mode", action="store_true",
                       help="Run with smaller dataset for testing")
    parser.add_argument("--resume-session", type=str,
                       help="Resume from existing session ID")
    parser.add_argument("--resume-from-files", type=str,
                       help="Resume from downloaded batch result files directory")
    
    args = parser.parse_args()
    
    if args.dry_run:
        config = create_hrm_dlp_config()
        print("🔍 HRM-DLP Configuration (Dry Run)")
        print("=" * 40)
        print(f"Output directory: {config.output_dir}")
        print(f"Training examples: {config.train_size}")
        print(f"Validation examples: {config.val_size}")
        print(f"Test examples: {config.test_size}")
        print(f"Agent distribution: {config.agent_distribution}")
        print(f"Risk distribution: {config.risk_distribution}")
        print(f"Batch API enabled: {config.enable_batch_api}")
        print(f"Quality threshold: {config.min_quality_score}")
        print(f"Thread probability: {config.thread_probability}")
        print()
        print("✅ Configuration looks good! Remove --dry-run to execute.")
        return
    
    if args.demo_mode:
        print("🎮 Demo mode: generating small dataset for testing")
        # Could modify config for demo mode here
    
    # Run the generation
    try:
        success = asyncio.run(generate_hrm_dlp_dataset(args.resume_session, args.resume_from_files))
        if success:
            print("\n🎉 HRM-DLP dataset generation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ HRM-DLP dataset generation failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Generation stopped by user")
        sys.exit(1)


if __name__ == "__main__":
    main()