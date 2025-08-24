#!/usr/bin/env python3
"""
Quick Start Script for HRM-DLP

This script demonstrates the complete pipeline:
1. Generate synthetic data
2. Train tokenizer  
3. Train model
4. Evaluate model

Usage:
    python quick_start_dlp.py --quick-demo          # Small demo (1000 samples, rule-based)
    python quick_start_dlp.py --quick-demo --llm    # Small demo with LLM generation
    python quick_start_dlp.py --full                # Full training (60k samples, rule-based)
    python quick_start_dlp.py --full --llm          # Full training with LLM generation
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import torch


def run_command(cmd: str, description: str = ""):
    """Run a command and handle errors"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Command failed after {duration:.1f}s")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: Completed in {duration:.1f}s")
        if result.stdout:
            print(f"Output: {result.stdout[:500]}...")
        return True


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "torch", "transformers", "wandb", "tqdm", "pydantic", 
        "omegaconf", "hydra-core", "sentencepiece", "adam-atan2"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    return True


def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/dlp_synth",
        "checkpoints",
        "results",
        "tokenizers"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def quick_demo(use_llm: bool = False):
    """Run a quick demo with your real dataset"""
    print(f"Starting HRM-DLP Quick Demo (using your real dataset)...")
    print("This will take approximately 20-30 minutes on a GPU")
    
    # 1. Use your existing real dataset
    data_dir = "data/runs/run_20250824_123640_a2f52bf9/split_outputs"
    
    print("Step 1: Using your existing real dataset")
    print(f"  - Dataset location: {data_dir}")
    print(f"  - Train file: val_examples_augmented.jsonl (using as train since train file is empty)")
    print(f"  - Val file: test_examples_augmented.jsonl") 
    print("  - This is high-quality LLM-generated data, no need to generate synthetic data!")
    
    # 2. Train model (simplified config)
    train_config = f"""# Quick Demo Training Configuration
data_path: {data_dir}
train_file: val_examples_augmented.jsonl
val_file: test_examples_augmented.jsonl
test_file: test_examples_augmented.jsonl
tokenizer_path: null
max_length: 512

arch:
  hidden_size: 256
  num_heads: 4
  H_layers: 2
  L_layers: 2
  H_cycles: 1
  L_cycles: 1
  expansion: 4.0
  pos_encodings: rope
  num_doc_labels: 4
  num_bio_tags: 21
  memory_dim: 128
  use_fusion_gates: true
  use_act: false
  forward_dtype: bfloat16

global_batch_size: 32
epochs: 1
lr: 1e-3
lr_min_ratio: 0.1
lr_warmup_steps: 100
weight_decay: 0.01
beta1: 0.9
beta2: 0.95

doc_loss_weight: 1.0
span_loss_weight: 1.0
mask_denoise_weight: 0.3
section_shuffle_weight: 0.2
label_smoothing: 0.05

eval_interval: 100
checkpoint_every_eval: true
project_name: hrm-dlp-demo
run_name: quick_demo
checkpoint_path: checkpoints/demo

seed: 42
num_workers: 2
"""
    
    # Save demo config
    with open("config/dlp_demo.yaml", "w") as f:
        f.write(train_config)
    
    success = run_command(
        "python pretrain_dlp.py --config-name dlp_demo",
        "Step 2: Training HRM-DLP model (quick demo)"
    )
    if not success:
        return False
    
    # 3. Evaluate model
    success = run_command(
        f"python evaluate_dlp.py --checkpoint checkpoints/demo/best_checkpoint.pt "
        f"--data-path {data_dir}/test.jsonl --output results/demo_results.json",
        "Step 3: Evaluating trained model"
    )
    if not success:
        return False
    
    print("\n🎉 Quick demo completed successfully!")
    print("Check the results in:")
    print("  - results/demo_results.json")
    print("  - checkpoints/demo/")
    print("  - Weights & Biases dashboard (if configured)")
    
    return True


def full_training(use_llm: bool = False):
    """Run full training pipeline"""
    data_type = "LLM-generated" if use_llm else "rule-based"
    print(f"Starting HRM-DLP Full Training ({data_type})...")
    print("This will take several hours on a GPU")
    
    # 1. Generate full synthetic dataset
    if use_llm:
        success = run_command(
            "python scripts/llm_data_generator.py --output-dir data/dlp_full_llm "
            "--train-size 60000 --val-size 5000 --test-size 5000 "
            "--llm-provider openai --model-name gpt-4o-mini --batch-size 10",
            "Step 1: Generating LLM-based DLP dataset (70k examples)"
        )
        data_dir = "data/dlp_full_llm"
    else:
        success = run_command(
            "python scripts/make_synth_data.py --output-dir data/dlp_full "
            "--train-size 60000 --val-size 5000 --test-size 5000 --seed 42",
            "Step 1: Generating rule-based DLP dataset (70k examples)"
        )
        data_dir = "data/dlp_full"
    if not success:
        return False
    
    # 2. Train tokenizer
    success = run_command(
        f"python -c \""
        f"from hrm_dlp.tokenizer import create_tokenizer; "
        f"tokenizer = create_tokenizer("
        f"['{data_dir}/train.jsonl', '{data_dir}/val.jsonl'], "
        f"'tokenizers/dlp_tokenizer', vocab_size=16000"
        f")\"",
        "Step 2: Training SentencePiece tokenizer"
    )
    if not success:
        print("Warning: Tokenizer training failed, will use simple tokenizer")
    
    # 3. Train model with full config
    success = run_command(
        f"python pretrain_dlp.py data_path={data_dir} "
        f"tokenizer_path=tokenizers/dlp_tokenizer.model "
        f"project_name=hrm-dlp-full run_name=full_training_{data_type.replace('-', '_')} "
        f"checkpoint_path=checkpoints/full",
        "Step 3: Training HRM-DLP model (full)"
    )
    if not success:
        return False
    
    # 4. Comprehensive evaluation
    success = run_command(
        f"python evaluate_dlp.py --checkpoint checkpoints/full/best_checkpoint.pt "
        f"--data-path {data_dir}/test.jsonl --output results/full_results.json",
        "Step 4: Comprehensive evaluation"
    )
    if not success:
        return False
    
    print("\n🎉 Full training completed successfully!")
    print("Check the results in:")
    print("  - results/full_results.json") 
    print("  - checkpoints/full/")
    print("  - Weights & Biases dashboard")
    
    return True


def show_example_usage():
    """Show example of how to use the trained model"""
    example_code = '''
# Example: Using trained HRM-DLP model for inference

import torch
from hrm_dlp.model import create_dlp_model  
from hrm_dlp.tokenizer import DLPTokenizer
from hrm_dlp.dsl import DSLSerializer

# Load model
checkpoint = torch.load("checkpoints/demo/best_checkpoint.pt")
model = create_dlp_model(checkpoint["config"]["arch"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer
tokenizer = DLPTokenizer("tokenizers/dlp_tokenizer.model")

# Create example email
email_data = {
    "channel": "email",
    "user": {"role": "LEGAL", "dept": "CORP"},
    "recipients": ["external@gmail.com"],
    "subject": "Confidential client data",
    "body": "Please find the credit card 4532 1234 5678 9012 for processing.",
    "attachments": [],
    "links": []
}

# Process through DSL
serializer = DSLSerializer()
result = serializer.serialize(email_data)
dsl_text = result.dsl_text

# Tokenize
token_ids = tokenizer.encode(dsl_text)
input_ids = torch.tensor([token_ids])

# Inference
with torch.no_grad():
    outputs = model(input_ids)
    
    # Document scores
    doc_probs = torch.sigmoid(outputs.doc_logits)
    print("Document scores:", doc_probs.numpy())
    
    # Span predictions
    span_preds = outputs.span_logits.argmax(dim=-1)
    print("Predicted spans:", span_preds.numpy())
    
    # Decision
    sensitive = doc_probs[0, 0] > 0.5  # sensitivity
    exposure = doc_probs[0, 1] > 0.5   # exposure risk
    
    if sensitive and exposure:
        print("DECISION: BLOCK - Sensitive data to external recipient")
    elif sensitive:
        print("DECISION: WARN - Sensitive data detected")
    else:
        print("DECISION: ALLOW")
'''
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    print(example_code)


def main():
    parser = argparse.ArgumentParser(
        description="HRM-DLP Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start_dlp.py --quick-demo         # Run 20-minute demo (rule-based)
  python quick_start_dlp.py --quick-demo --llm   # Run demo with LLM generation
  python quick_start_dlp.py --full               # Full training (rule-based)
  python quick_start_dlp.py --full --llm         # Full training with LLM generation
  python quick_start_dlp.py --check              # Check requirements only
        """
    )
    
    parser.add_argument("--quick-demo", action="store_true",
                       help="Run quick demo with small dataset")
    parser.add_argument("--full", action="store_true", 
                       help="Run full training pipeline")
    parser.add_argument("--llm", action="store_true",
                       help="Use LLM for data generation (requires API key)")
    parser.add_argument("--check", action="store_true",
                       help="Check requirements and setup")
    parser.add_argument("--show-example", action="store_true",
                       help="Show example usage code")
    
    args = parser.parse_args()
    
    if not any([args.quick_demo, args.full, args.check, args.show_example]):
        parser.print_help()
        return
    
    # Add HRM directory to path for imports  
    sys.path.insert(0, str(Path(__file__).parent.parent / "HRM"))
    
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    if args.show_example:
        show_example_usage()
        return
    
    if args.check or args.quick_demo or args.full:
        print("HRM-DLP Quick Start")
        print("="*60)
        
        # Check requirements
        print("\nChecking requirements...")
        if not check_requirements():
            print("❌ Requirements check failed")
            return
        print("✅ Requirements check passed")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available, training will be slow")
        
        # Setup directories
        print("\nSetting up directories...")
        setup_directories()
        print("✅ Directories created")
        
        if args.check:
            print("\n✅ Setup check completed successfully")
            return
    
    # Check for LLM requirements
    if args.llm and (args.quick_demo or args.full):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ LLM generation requires OPENAI_API_KEY environment variable")
            print("Set it with: export OPENAI_API_KEY=your_api_key_here")
            return
        print("✅ OpenAI API key found")
    
    # Run requested pipeline
    if args.quick_demo:
        success = quick_demo(use_llm=args.llm)
        if success:
            show_example_usage()
    elif args.full:
        success = full_training(use_llm=args.llm)
        if success:
            show_example_usage()


if __name__ == "__main__":
    main()