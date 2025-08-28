#!/usr/bin/env python3
"""
Clean up and organize HRM-DLP project data files.
"""

import shutil
from pathlib import Path
from datetime import datetime

def cleanup_project():
    """Clean up and organize project data files."""
    
    print("🧹 HRM-DLP Project Cleanup")
    print("=" * 50)
    
    # Base paths
    base_dir = Path("/Users/guynachshon/Documents/baddon-ai/labs/HRM-DLP")
    scripts_dir = base_dir / "HRM" / "scripts" / "agentic_data_generator"
    
    # Create archive directory for old data
    archive_dir = base_dir / "data" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Create final structure
    final_dir = base_dir / "data" / "hrm_dlp_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Base directory: {base_dir}")
    print(f"📁 Archive directory: {archive_dir}")
    print(f"📁 Final data directory: {final_dir}")
    print()
    
    # 1. Identify the final/best training data
    print("🔍 Step 1: Identifying final training data...")
    
    # The best data is in /data/training_data/ (copied by our final processing)
    best_training_data = base_dir / "data" / "training_data"
    if best_training_data.exists():
        print(f"✅ Found final training data: {best_training_data}")
        
        # Copy to final location
        for file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            src = best_training_data / file
            dst = final_dir / file
            if src.exists():
                shutil.copy2(src, dst)
                
                # Check file stats
                with open(dst, 'r') as f:
                    count = sum(1 for _ in f)
                size = dst.stat().st_size
                print(f"   ✅ {file}: {count} examples ({size:,} bytes)")
            else:
                print(f"   ❌ {file}: Not found")
                
        print(f"📁 Final training data → {final_dir}")
    else:
        print("❌ No final training data found")
    
    print()
    
    # 2. Archive old runs and intermediate data
    print("🗃️  Step 2: Archiving old runs and intermediate data...")
    
    # Archive script data/runs (successful session data)
    scripts_data_runs = scripts_dir / "data" / "runs"
    if scripts_data_runs.exists():
        archive_runs = archive_dir / "script_runs"
        if archive_runs.exists():
            shutil.rmtree(archive_runs)
        shutil.move(str(scripts_data_runs), str(archive_runs))
        print(f"   📦 Moved script runs → {archive_runs}")
    
    # Archive main data/runs
    main_data_runs = base_dir / "data" / "runs"  
    if main_data_runs.exists():
        archive_main_runs = archive_dir / "main_runs"
        if archive_main_runs.exists():
            shutil.rmtree(archive_main_runs)
        shutil.move(str(main_data_runs), str(archive_main_runs))
        print(f"   📦 Moved main runs → {archive_main_runs}")
    
    # Archive resume data
    resume_dir = base_dir / "data" / "resume"
    if resume_dir.exists():
        archive_resume = archive_dir / "resume"
        if archive_resume.exists():
            shutil.rmtree(archive_resume)
        shutil.move(str(resume_dir), str(archive_resume))
        print(f"   📦 Moved resume data → {archive_resume}")
        
    # Archive existing archive if it has old formats
    existing_archive = base_dir / "data" / "archive"
    old_formats_dir = archive_dir / "old_formats"
    for old_format in ["dlp_ag", "dlp_agentic", "dlp_synth", "processed"]:
        old_path = existing_archive / old_format
        if old_path.exists():
            new_path = old_formats_dir / old_format
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                shutil.rmtree(new_path)
            shutil.move(str(old_path), str(new_path))
            print(f"   📦 Moved old format {old_format} → {new_path}")
    
    print()
    
    # 3. Keep the enhanced data as reference
    print("📊 Step 3: Organizing enhanced data...")
    
    # Scripts enhanced data (our final generated data)
    scripts_enhanced = scripts_dir / "data" / "hrm_dlp_enhanced"
    if scripts_enhanced.exists():
        reference_dir = final_dir / "reference_enhanced"
        if reference_dir.exists():
            shutil.rmtree(reference_dir)
        shutil.copytree(scripts_enhanced, reference_dir)
        print(f"   ✅ Copied enhanced reference → {reference_dir}")
        
        # Show stats
        for file in reference_dir.glob("*.jsonl"):
            with open(file, 'r') as f:
                count = sum(1 for _ in f)
            size = file.stat().st_size
            print(f"      {file.name}: {count} examples ({size:,} bytes)")
    
    # Main enhanced data (if different)
    main_enhanced = base_dir / "data" / "hrm_dlp_enhanced"
    if main_enhanced.exists() and main_enhanced != scripts_enhanced:
        main_ref_dir = final_dir / "main_enhanced"
        if main_ref_dir.exists():
            shutil.rmtree(main_ref_dir)
        shutil.copytree(main_enhanced, main_ref_dir)
        print(f"   ✅ Copied main enhanced → {main_ref_dir}")
    
    print()
    
    # 4. Clean up temporary and test files
    print("🗑️  Step 4: Removing temporary files...")
    
    temp_files_removed = 0
    
    # Remove test and debug files from scripts directory
    for pattern in ["test_*.py", "debug_*.py", "verify_*.py", "*_test.py", "resume_*.py", "complete_*.py", "final_*.py"]:
        for file in scripts_dir.glob(pattern):
            if file.is_file():
                file.unlink()
                temp_files_removed += 1
                print(f"   🗑️  Removed {file.name}")
    
    # Remove .pyc files and __pycache__
    for pycache in scripts_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            temp_files_removed += 1
            print(f"   🗑️  Removed __pycache__")
    
    # Remove openai-batches.json if exists
    batch_log = scripts_dir / "openai-batches.json"
    if batch_log.exists():
        batch_log.unlink()
        temp_files_removed += 1
        print(f"   🗑️  Removed batch log")
    
    print(f"   📊 Removed {temp_files_removed} temporary files")
    print()
    
    # 5. Create final summary
    print("📋 Step 5: Final organization summary...")
    
    # Create README for final data
    readme_content = f"""# HRM-DLP Training Dataset
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Final Training Data
- `train.jsonl`: Training examples with rich attachment metadata
- `val.jsonl`: Validation examples  
- `test.jsonl`: Test examples

## Format
Each example contains:
- Rich attachment objects with sensitivity_indicators, content_summary, mime_type, size
- DLP format with spans, labels, and context extraction
- Proper augmentation applied (2.0x ratio)

## Reference
- `reference_enhanced/`: Raw enhanced examples before quality filtering
- `../archive/`: Archived runs and intermediate data

Ready for HRM-DLP model training!
"""
    
    readme_file = final_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_file}")
    print()
    
    # Final stats
    print("📊 FINAL PROJECT ORGANIZATION:")
    print("=" * 50)
    print(f"🎯 Training Data: {final_dir}")
    
    total_examples = 0
    total_size = 0
    for file in final_dir.glob("*.jsonl"):
        with open(file, 'r') as f:
            count = sum(1 for _ in f)
        size = file.stat().st_size
        total_examples += count
        total_size += size
        print(f"   ✅ {file.name}: {count} examples ({size:,} bytes)")
    
    print(f"   📊 TOTAL: {total_examples} examples ({total_size:,} bytes)")
    print()
    print(f"📦 Archived Data: {archive_dir}")
    print(f"📁 Project Structure: Clean and organized")
    print()
    print("🎉 Cleanup completed successfully!")
    print(f"🚀 Ready to train HRM-DLP model with data in: {final_dir}")

if __name__ == "__main__":
    try:
        cleanup_project()
        print("\n✅ Project cleanup completed!")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()