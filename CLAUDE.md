# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the Hierarchical Reasoning Model (HRM) adapted for Data Loss Prevention (DLP). The codebase combines:

1. **Base HRM Model** (`/HRM/`) - Original hierarchical reasoning architecture for puzzle solving
2. **DLP Adaptation** - Extensions for email/chat content analysis, PII detection, and trust scoring

## Key Commands

### Environment Setup
```bash
# Install dependencies (from HRM directory)
cd HRM
pip install -r requirements.txt

# For CUDA extensions (if needed)
pip install flash-attn  # For Ampere+ GPUs
# OR for Hopper GPUs:
# git clone git@github.com:Dao-AILab/flash-attention.git
# cd flash-attention/hopper && python setup.py install
```

### Dataset Generation

#### HRM Puzzle Datasets
```bash
cd HRM

# ARC datasets
python dataset/build_arc_dataset.py  # ARC-1
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Sudoku datasets  
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze datasets
python dataset/build_maze_dataset.py
```

#### DLP Synthetic Data Generation
```bash
cd HRM

# Quick demo dataset (1,400 examples)
python quick_start_dlp.py --quick-demo

# Full synthetic dataset (70k examples)
python scripts/make_synth_data.py --output-dir data/dlp_synth --train-size 60000 --val-size 5000 --test-size 5000 --seed 42

# Agentic data generation with batch processing (advanced)
python scripts/agentic_data_generator.py --target 50000 --output data/dlp_agentic --enable-dashboard
```

### Training Commands

#### HRM Puzzle Training
```bash
cd HRM
# Single GPU Training
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU Training  
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py [additional_args]
```

#### DLP Training
```bash
cd HRM
# DLP model training with synthetic data
python pretrain_dlp.py data_path=data/dlp_synth global_batch_size=768 epochs=2 lr=3e-4

# Quick DLP demo training (20 minutes)
python quick_start_dlp.py --quick-demo
```

### Evaluation
```bash
cd HRM
# HRM puzzle evaluation
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# DLP evaluation
python evaluate_dlp.py --checkpoint checkpoints/best_checkpoint.pt --data-path data/dlp_synth/test.jsonl --output results/evaluation.json
```

## Architecture Overview

### HRM Core Model
- **Hierarchical Structure**: Two-tier architecture with fast (detailed) and slow (abstract) reasoning modules
- **Fast Module**: Transformer encoder with RoPE, GLU, RMSNorm (d=384, L=8, H=6)
- **Slow Module**: Recurrent encoder over 64-token segments with learned fusion gates
- **ACT (Adaptive Compute Time)**: Optional halting mechanism using Q-learning (disabled for deterministic latency in DLP)

### DLP Extensions (Planned Architecture)
- **Input Processing**: DSL serialization for email/chat/PR content with role, recipients, metadata
- **Output Heads**:
  - Document-level scores: Sensitivity, ExposureRisk, ContextConsistency, ObfuscationRisk
  - Token-level BIO tags: EMAIL, PHONE, PAN, SSN, SECRET_KEY, DB_URI, NDA_TERM, etc.
  - 256-D memory summary vectors
- **Multi-task Loss**: BCE(doc) + CE(BIO) + auxiliary mask-denoise + section-shuffle objectives

### Key Components

#### Models (`HRM/models/`)
- `hrm/hrm_act_v1.py` - Main HRM architecture with ACT
- `layers.py` - Transformer building blocks (Attention, SwiGLU, etc.)
- `losses.py` - ACT loss implementation
- `sparse_embedding.py` - Distributed sparse embeddings

#### Training (`HRM/`)
- `pretrain.py` - Main training script with Hydra configuration
- `pretrain_dlp.py` - DLP-specific training script
- `evaluate.py` - Model evaluation with metric computation
- `evaluate_dlp.py` - DLP-specific evaluation script
- `puzzle_dataset.py` - Dataset loading and batching
- `quick_start_dlp.py` - Complete DLP demo pipeline

#### DLP Components (`HRM/hrm_dlp/`)
- `model.py` - HRM-DLP architecture with multi-task heads
- `dataset.py` - DLP data loading and preprocessing
- `losses.py` - Multi-task loss functions
- `dsl.py` - DSL serialization for email/chat content
- `tokenizer.py` - SentencePiece tokenizer

#### Data Generation (`HRM/scripts/`)
- `make_synth_data.py` - Basic synthetic DLP data generator
- `agentic_data_generator.py` - Advanced 3-tier agentic system with batch processing
- `batch_tracker.py` - Persistent batch state management
- `batch_monitor.py` - Real-time batch monitoring system
- `task_dashboard.py` - Live terminal dashboard for generation progress
- `business_context.py` - Business domain context analysis
- `semantic_obfuscation.py` - Advanced obfuscation techniques
- `recover_checkpoint_data.py` - Recovery utilities for interrupted runs

#### Configuration (`HRM/config/`)
- `cfg_pretrain.yaml` - Training hyperparameters
- `dlp_train.yaml` - DLP-specific training configuration
- `arch/hrm_v1.yaml` - Model architecture settings

## Data Handling

### Current Datasets (HRM)
- **ARC-AGI**: Abstract reasoning tasks with visual grids
- **Sudoku**: 9x9 puzzle solving (extreme difficulty variants)
- **Maze**: Path finding in 30x30 grids

### DLP Dataset Schema
```json
{
  "channel": "email|chat|pr|upload",
  "user": {"role": "LEGAL", "dept": "CORP", "seniority": "SENIOR"},
  "recipients": ["external@domain.com"],
  "subject": "Email subject",
  "body": "Email content with PII...",
  "labels": {"sensitivity": 1, "exposure": 0, "context": 1},
  "spans": [{"type": "PAN", "start": 128, "end": 147}]
}
```

### Agentic Data Generation System

The advanced data generation system uses a 3-tier architecture:

#### Tier 1: Manager Agent
- Controls dataset balance and quality
- Manages batch processing across OpenAI/Anthropic APIs
- Persistent checkpoint system for recovery from interruptions
- Live dashboard with real-time progress tracking

#### Tier 2: Specialized Agents  
- **Legal Agent**: Contracts, NDAs, legal correspondence
- **Finance Agent**: Payment processing, financial records
- **HR Agent**: Employee data, benefits, personnel files
- **Security Agent**: Incident reports, access logs
- **Casual Agent**: Personal emails mixed in corporate context

#### Tier 3: Conversational Agents
- Multi-turn conversation simulation
- Context-aware follow-up messages
- Realistic thread progression

#### Batch Processing Architecture
- **BatchTracker**: Persistent state management with 24-hour timeouts
- **BatchMonitor**: Real-time monitoring with automatic fallback to concurrent processing
- **TaskDashboard**: Live terminal UI showing batch status, progress, and thread activity
- **Recovery System**: Automatic checkpoint recovery for interrupted runs

#### Key Features
- Supports both OpenAI Batch API and Anthropic Message Batches API
- Automatic fallback to concurrent processing for small batches or timeouts
- Business context validation for realistic scenario generation
- Semantic obfuscation techniques (base64, homoglyphs, zero-width chars)
- Thread-safe progress tracking across multiple concurrent operations

## Development Workflows

### Adding New Puzzle Types
1. Create dataset builder in `dataset/build_<puzzle>_dataset.py`
2. Follow existing patterns from `build_sudoku_dataset.py` or `build_arc_dataset.py`
3. Implement data loading in `puzzle_dataset.py`
4. Update configuration files as needed

### DLP Development Workflows

#### Adding New Domain Agents
1. Extend the agent system in `scripts/agentic_data_generator.py`
2. Add domain-specific prompts and context patterns
3. Update business context validation in `scripts/business_context.py`
4. Test with small batches before full generation

#### Batch Processing Development
```bash
cd HRM
# Test batch processing with real API calls
python scripts/test_batch_apis.py

# Test progress indicators and dashboard
python scripts/test_batch_progress.py

# Test batch recovery from interruptions
python scripts/recover_checkpoint_data.py --checkpoint-dir checkpoints/data_generation
```

#### Data Quality Validation
```bash
cd HRM
# Analyze generated data quality
python scripts/business_context.py --validate data/dlp_agentic/train.jsonl

# Test semantic obfuscation patterns
python scripts/semantic_obfuscation.py --test-patterns
```

### Model Architecture Changes
1. Modify core architecture in `models/hrm/`
2. For DLP-specific changes, work in `hrm_dlp/model.py`
3. Update configuration schemas in `config/arch/` or `config/dlp_train.yaml`
4. Adjust loss functions in `models/losses.py` or `hrm_dlp/losses.py`
5. Test with small datasets before full training

### Performance Optimization
- Model uses fixed compute budget for deterministic latency
- Training supports mixed precision (bf16) and gradient accumulation
- ONNX export available for production deployment
- Multi-GPU training via torchrun

## Monitoring and Evaluation

### Key Metrics
- **Exact Accuracy**: Primary evaluation metric (check `eval/exact_accuracy` in W&B)
- **Training Stability**: Early stopping recommended when accuracy approaches 100%
- **Numerical Stability**: Watch for Q-learning instabilities in late-stage overfitting

### Weights & Biases Integration
```bash
wandb login  # Required for experiment tracking
```

All training runs automatically log metrics, model checkpoints, and hyperparameters to W&B.

## Important Notes

### General Development
- Small-sample learning exhibits ±2 point accuracy variance
- For Sudoku-Extreme datasets, use early stopping to avoid numerical instability
- CUDA extensions required for optimal performance
- Configuration uses Hydra for flexible parameter management
- The codebase supports both research (ACT enabled) and production (deterministic) modes

### DLP-Specific Considerations
- **API Keys Required**: Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables for agentic data generation
- **Batch Processing**: OpenAI and Anthropic batch APIs have different rate limits and processing times
- **Data Recovery**: The system creates automatic checkpoints - use recovery utilities if generation is interrupted
- **Dashboard Display**: The live dashboard requires a terminal that supports rich formatting (most modern terminals)
- **Thread Safety**: All dashboard and batch operations are thread-safe for concurrent processing
- **Checkpoint Management**: Large data generation runs create substantial checkpoint files in `checkpoints/data_generation/`

### Environment Variables
```bash
# Required for agentic data generation
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: Weights & Biases for experiment tracking
export WANDB_API_KEY="your-wandb-key"
```

### Monitoring Long-Running Jobs
- Use `--enable-dashboard` flag for live progress monitoring
- Checkpoint files are saved every 50 examples for recovery
- Batch processing automatically falls back to concurrent mode on timeouts
- Use `Ctrl+C` to gracefully stop generation (preserves checkpoints)