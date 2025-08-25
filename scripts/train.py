#!/usr/bin/env python3
"""
Unified Training Script for HRM-DLP

Supports training both HRM puzzle models and DLP models with:
- Automatic configuration loading from YAML files
- Multi-GPU training with distributed support
- Comprehensive logging and checkpointing
- Model-specific optimizations
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hrm_core.models import HRMBase, HRMConfig
from dlp.models import DLPModel, DLPModelConfig, create_dlp_model
from dlp.data import create_dlp_dataloader, DLPDatasetConfig, create_tokenizer
from dlp.training import DLPTrainer, DLPTrainingConfig, create_dlp_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and merge configuration files"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config and merge
    base_config_path = project_root / "config" / "base.yaml"
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Deep merge configurations
        config = merge_configs(base_config, config)
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_distributed():
    """Setup distributed training if available"""
    if "WORLD_SIZE" in os.environ:
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        # Set device for this process
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        return True, local_rank
    else:
        return False, 0


def train_dlp_model(config: Dict[str, Any], args: argparse.Namespace):
    """Train a DLP model"""
    print("Training DLP model...")
    
    # Setup distributed training
    is_distributed, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load model configuration
    model_config_path = project_root / config.get("model_config", "config/models/dlp.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Create model
    dlp_config = DLPModelConfig(**model_config["model"])
    model = DLPModel(dlp_config)
    model.to(device)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Create tokenizer
    tokenizer = create_tokenizer(
        vocab_size=dlp_config.vocab_size,
        model_path=config["data"].get("tokenizer_path")
    )
    
    # Create datasets
    dataset_config = DLPDatasetConfig(
        max_length=config["data"]["max_length"],
        doc_labels=model_config.get("doc_labels", ["sensitivity", "exposure", "context", "obfuscation"])
    )
    
    # Determine data paths with fallback logic
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    
    # Check if primary paths exist and have content
    if not Path(train_path).exists() or Path(train_path).stat().st_size == 0:
        print(f"Primary train path {train_path} not found or empty, using fallback")
        train_path = config["data"].get("fallback_train_path", train_path)
    
    if not Path(val_path).exists() or Path(val_path).stat().st_size == 0:
        print(f"Primary val path {val_path} not found or empty, using fallback")
        val_path = config["data"].get("fallback_val_path", val_path)
    
    # Create data loaders
    train_dataloader = create_dlp_dataloader(
        train_path,
        tokenizer,
        dataset_config,
        batch_size=config["training"]["per_device_batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = create_dlp_dataloader(
        val_path,
        tokenizer,
        dataset_config,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training dataset size: {len(train_dataloader.dataset)}")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")
    
    # Create loss function
    loss_fn = create_dlp_loss(
        vocab_size=dlp_config.vocab_size,
        config_dict=model_config.get("loss", {}),
        num_doc_labels=dlp_config.num_doc_labels,
        num_bio_tags=dlp_config.num_bio_tags,
        hidden_size=dlp_config.hidden_size
    )
    
    # Create trainer configuration
    training_config = DLPTrainingConfig(
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        global_batch_size=config["training"]["per_device_batch_size"] * config["training"]["gradient_accumulation_steps"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        lr_warmup_steps=config["training"]["lr_warmup_steps"],
        lr_min_ratio=config["training"]["lr_min_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        use_amp=config["training"]["use_amp"],
        amp_dtype=config["training"]["amp_dtype"],
        checkpoint_dir=config["output"]["checkpoint_dir"],
        save_every_n_steps=config["training"]["save_every_n_steps"],
        eval_every_n_steps=config["training"]["eval_every_n_steps"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_metric=config["training"]["early_stopping_metric"],
        early_stopping_mode=config["training"]["early_stopping_mode"],
        project_name=config["experiment"]["project_name"],
        run_name=config["experiment"].get("run_name"),
        use_wandb=config.get("logging", {}).get("use_wandb", True)
    )
    
    # Create trainer
    trainer = DLPTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        config=training_config,
        device=device
    )
    
    # Start training
    training_stats = trainer.train()
    
    # Save final results
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"Training completed! Results saved to {results_dir}")
    return training_stats


def train_hrm_model(config: Dict[str, Any], args: argparse.Namespace):
    """Train an HRM model (placeholder for puzzle training)"""
    print("HRM puzzle training not yet implemented in unified script")
    print("Use legacy scripts in HRM/ directory for puzzle training")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Unified HRM-DLP Training Script")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dlp", "hrm"],
        default="dlp",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    if args.dry_run:
        print("Configuration validation successful!")
        print(f"Model type: {args.model_type}")
        print(f"Configuration keys: {list(config.keys())}")
        return 0
    
    # Set random seed for reproducibility
    seed = config.get("system", {}).get("seed", 42)
    torch.manual_seed(seed)
    
    # Create output directories
    for dir_key in ["checkpoint_dir", "results_dir", "logs_dir"]:
        if dir_key in config.get("output", {}):
            Path(config["output"][dir_key]).mkdir(parents=True, exist_ok=True)
    
    # Train model based on type
    try:
        if args.model_type == "dlp":
            training_stats = train_dlp_model(config, args)
        elif args.model_type == "hrm":
            training_stats = train_hrm_model(config, args)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        print("Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())