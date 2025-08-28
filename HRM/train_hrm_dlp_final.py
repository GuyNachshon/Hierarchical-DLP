#!/usr/bin/env python3
"""
HRM-DLP Final Training Script

Train the HRM-DLP model on the clean, high-quality dataset with rich attachment metadata.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the HRM directory to Python path
hrm_dir = Path(__file__).parent
sys.path.insert(0, str(hrm_dir))

def main():
    """Launch HRM-DLP training with the final clean dataset."""
    
    print("🚀 Starting HRM-DLP Final Training")
    print("=" * 50)
    
    # Check dataset exists
    data_path = hrm_dir / "../data/hrm_dlp_final"
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        print("Please ensure the clean dataset is available.")
        sys.exit(1)
    
    # Check dataset files
    required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    for file in required_files:
        file_path = data_path / file
        if not file_path.exists():
            print(f"❌ Missing dataset file: {file}")
            sys.exit(1)
        
        # Show file stats
        with open(file_path, 'r') as f:
            count = sum(1 for _ in f)
        size = file_path.stat().st_size
        print(f"✅ {file}: {count} examples ({size:,} bytes)")
    
    print()
    
    # Configuration
    config_path = hrm_dir / "config" / "hrm_dlp_final_train.yaml"
    print(f"📋 Using configuration: {config_path}")
    
    # Load and display key config settings
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("📊 Training Configuration:")
    print(f"   • Dataset: {config['data_path']}")
    print(f"   • Batch size: {config['global_batch_size']}")
    print(f"   • Epochs: {config['epochs']}")
    print(f"   • Learning rate: {config['lr']}")
    print(f"   • Model: {config['arch'].get('name', 'HRM_DLP')}")
    print()
    
    # Create checkpoints directory
    checkpoint_dir = hrm_dir / config.get('checkpoint_path', 'checkpoints/hrm_dlp_final')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    print()
    
    # Launch training using pretrain.py with Hydra
    print("🔥 Launching training...")
    print("-" * 30)
    
    # Set working directory and run
    os.chdir(hrm_dir)
    
    # Use Hydra to launch training with the configuration
    import subprocess
    
    cmd = [
        sys.executable, "pretrain.py",
        f"--config-path=config",
        f"--config-name=hrm_dlp_final_train",
        f"hydra.run.dir={checkpoint_dir}",
        f"hydra.job.chdir=true"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, cwd=hrm_dir)
        
        print()
        print("🎉 Training completed successfully!")
        print(f"📁 Checkpoints saved to: {checkpoint_dir}")
        
        # Check for final model
        final_model = checkpoint_dir / "model_final.pt"
        if final_model.exists():
            print(f"✅ Final model: {final_model}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("Check the logs above for error details.")
        return False
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return False

def check_requirements():
    """Check if required dependencies are available."""
    required_packages = [
        ('torch', 'torch'),
        ('tqdm', 'tqdm'), 
        ('wandb', 'wandb'),
        ('hydra-core', 'hydra'),
        ('omegaconf', 'omegaconf'),
        ('pydantic', 'pydantic')
    ]
    missing = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   • {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    print("HRM-DLP Final Training")
    print("Training on clean dataset with rich attachment metadata")
    print()
    
    # Check requirements first
    if not check_requirements():
        sys.exit(1)
    
    # Run training
    success = main()
    
    if success:
        print("\n✅ HRM-DLP training completed!")
        print("🚀 Model ready for evaluation and deployment")
    else:
        print("\n❌ Training failed")
        sys.exit(1)