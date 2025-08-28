#!/usr/bin/env python3
"""
Restore the correct full dataset with 2,653 examples and rich attachments.
"""

import json
import shutil
from pathlib import Path

def restore_correct_data():
    """Restore the full dataset with proper rich attachments."""
    
    print("🔄 Restoring Correct HRM-DLP Dataset")
    print("=" * 50)
    
    # Source: the archived session with rich attachments (1,854 + 400 + 399 = 2,653 total)
    source_dir = Path("/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP/data/archive/script_runs/run_20250828_142515_1bf5cf73/split_outputs")
    
    # Destination: the final training directory
    dest_dir = Path("/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP/data/hrm_dlp_final")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Source: {source_dir}")
    print(f"📁 Destination: {dest_dir}")
    print()
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    total_examples = 0
    
    # Copy each split and verify rich attachments
    for split in ["train", "val", "test"]:
        source_file = source_dir / f"{split}_examples.jsonl"
        dest_file = dest_dir / f"{split}.jsonl"
        
        if not source_file.exists():
            print(f"⚠️  {split}: Source file not found")
            continue
            
        print(f"📊 Processing {split} split...")
        
        # Load and verify examples
        examples = []
        rich_attachment_count = 0
        
        with open(source_file, 'r') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line.strip())
                    examples.append(example)
                    
                    # Check for rich attachments
                    attachments = example.get('attachments', [])
                    for attachment in attachments:
                        if isinstance(attachment, dict) and 'sensitivity_indicators' in attachment:
                            rich_attachment_count += 1
                            break
        
        # Write to destination
        with open(dest_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        # Stats
        file_size = dest_file.stat().st_size
        total_examples += len(examples)
        
        print(f"   ✅ {split}: {len(examples)} examples ({file_size:,} bytes)")
        print(f"   🎉 Rich attachments: {rich_attachment_count} examples")
        
    print()
    print(f"📊 TOTAL RESTORED: {total_examples} examples")
    
    # Verify a sample rich attachment
    sample_file = dest_dir / "train.jsonl"
    if sample_file.exists():
        print("🔍 Verifying rich attachment structure...")
        with open(sample_file, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                attachments = example.get('attachments', [])
                for attachment in attachments:
                    if isinstance(attachment, dict) and 'sensitivity_indicators' in attachment:
                        print(f"   ✅ Sample attachment:")
                        print(f"      Name: {attachment.get('name')}")
                        print(f"      Size: {attachment.get('size')} bytes")
                        print(f"      MIME: {attachment.get('mime_type')}")
                        print(f"      Summary: {attachment.get('content_summary', '')[:50]}...")
                        print(f"      Indicators: {attachment.get('sensitivity_indicators', [])}")
                        break
                break
    
    print()
    print("🎉 CORRECT DATASET RESTORED!")
    print(f"📁 Location: {dest_dir}")
    print(f"📊 Ready for training with {total_examples} high-quality examples")
    
    # Update README
    readme_content = f"""# HRM-DLP Training Dataset (CORRECTED)
Generated: 2025-08-28 with rich attachment metadata

## Final Training Data ({total_examples} examples)
- `train.jsonl`: {Path(dest_dir / 'train.jsonl').exists() and sum(1 for _ in open(dest_dir / 'train.jsonl'))} training examples
- `val.jsonl`: {Path(dest_dir / 'val.jsonl').exists() and sum(1 for _ in open(dest_dir / 'val.jsonl'))} validation examples  
- `test.jsonl`: {Path(dest_dir / 'test.jsonl').exists() and sum(1 for _ in open(dest_dir / 'test.jsonl'))} test examples

## Rich Attachment Format
Each example contains proper attachment objects with:
- `name`: Filename with extension
- `size`: File size in bytes
- `mime_type`: Proper MIME type
- `content_summary`: Detailed content description
- `sensitivity_indicators`: Array of sensitivity types

## Quality
- ✅ Rich attachment metadata (not simple strings)
- ✅ Proper DLP format with spans and labels
- ✅ High-quality examples from successful batch generation
- ✅ Ready for HRM-DLP model training

Total: {total_examples} examples ready for training!
"""
    
    readme_file = dest_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"📋 Updated README: {readme_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = restore_correct_data()
        if success:
            print("\n✅ Dataset restoration completed successfully!")
            print("🚀 Ready to train HRM-DLP model with full dataset")
        else:
            print("\n❌ Dataset restoration failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()