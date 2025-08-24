# HRM-DLP: Hierarchical Reasoning Model for Data Loss Prevention

A clean, simplified implementation of the Hierarchical Reasoning Model (HRM) with extensions for Data Loss Prevention (DLP) tasks.

## Features

- **HRM Core**: Two-tier hierarchical architecture for sequential reasoning tasks
- **DLP Extension**: Email/chat content analysis, PII detection, and trust scoring
- **Simplified Architecture**: Clean codebase following KISS and clean code principles
- **Unified Interface**: Single entry point for training, evaluation, and data generation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd HRM-DLP

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo

```bash
# DLP demo (synthetic data)
python scripts/quick_start.py --demo-type dlp --quick

# HRM Sudoku demo
python scripts/quick_start.py --demo-type hrm-sudoku --quick
```

### 3. Generate Data

```bash
# Generate DLP synthetic data
python scripts/generate_data.py --type dlp --output-dir ./data/processed/dlp --train-size 1000

# Generate Sudoku puzzles
python scripts/generate_data.py --type sudoku --output-dir ./data/processed/sudoku --num-examples 1000
```

### 4. Train Models

```bash
# Train HRM on Sudoku
python scripts/train_hrm.py --dataset sudoku --data-path ./data/processed/sudoku --epochs 100

# Train DLP model
python scripts/train_dlp.py --data-path ./data/processed/dlp --epochs 10
```

### 5. Evaluate

```bash
# Evaluate HRM model
python scripts/evaluate.py --model-type hrm --checkpoint <checkpoint-path> --data-path <data-path>

# Evaluate DLP model  
python scripts/evaluate.py --model-type dlp --checkpoint <checkpoint-path> --data-path <data-path>
```

## Project Structure

```
HRM-DLP/
├── src/                    # Main source code
│   ├── hrm/               # Core HRM model
│   ├── dlp/               # DLP extensions
│   ├── data/              # Data generation
│   └── utils/             # Shared utilities
├── scripts/               # Entry point scripts
├── config/                # Configuration files
├── data/                  # Data storage
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Architecture

### HRM Core
- **Fast Module**: Transformer encoder for detailed computations (d=384, L=8, H=6)
- **Slow Module**: Recurrent encoder for abstract planning over 64-token segments
- **ACT**: Adaptive Compute Time with Q-learning (optional)

### DLP Extensions
- **Multi-task Heads**: Document-level scores + token-level BIO tags
- **PII Detection**: EMAIL, PHONE, PAN, SSN, SECRET_KEY, etc.
- **Memory Vectors**: 256-D summary representations
- **DSL Serialization**: Structured input format for email/chat content

## Configuration

Models are configured via YAML files in the `config/` directory:

- `config/hrm.yaml`: HRM model settings
- `config/dlp.yaml`: DLP model settings

## Contributing

This is a cleaned and simplified version of the original complex multi-agent system. The focus is on maintainability and clarity while preserving all core functionality.

## License

See LICENSE file for details.

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```