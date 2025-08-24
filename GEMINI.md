# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the HRM-DLP project.

## Project Overview

This project implements a Hierarchical Reasoning Model (HRM) for Data Loss Prevention (DLP). The model is designed to analyze text from various sources (e.g., emails, chat messages) and identify potential data leaks. It's a multi-task model that performs:

*   **Document-level classification:** Assigns scores for sensitivity, exposure, context, and obfuscation.
*   **Token-level BIO span tagging:** Identifies and classifies sensitive data spans (e.g., PII, secrets).
*   **Memory summary generation:** Creates a vector representation of the document for future use.

The project is built using Python and PyTorch. It includes scripts for synthetic data generation, model training, evaluation, and a quick start guide.

## Key Files

*   `HRM/hrm_dlp/model.py`: Defines the `HRMDLPModel` class, which is the core of the project. It adapts the original HRM architecture for DLP tasks.
*   `HRM/hrm_dlp/dataset.py`: Contains the `DLPDataset` class for loading and preprocessing the training data. It handles the JSONL format and BIO tag alignment.
*   `HRM/pretrain_dlp.py`: The main training script for the HRM-DLP model. It uses Hydra for configuration and Weights & Biases for experiment tracking.
*   `HRM/evaluate_dlp.py`: A script for evaluating the trained model. It computes a comprehensive set of metrics for both document classification and span tagging.
*   `HRM/quick_start_dlp.py`: A quick start script that demonstrates the entire pipeline, from data generation to model evaluation.
*   `HRM/scripts/make_synth_data.py`: A script for generating synthetic DLP training data. It creates realistic email/chat scenarios with embedded sensitive data.
*   `HRM/config/dlp_train.yaml`: The main configuration file for training the model.

## Building and Running

### 1. Installation

The project uses Python 3.12. The required packages are listed in `HRM/requirements.txt`. They can be installed using pip:

```bash
pip install -r HRM/requirements.txt
```

### 2. Data Generation

To generate the synthetic dataset, run the `make_synth_data.py` script:

```bash
python HRM/scripts/make_synth_data.py
```

This will create a `data/dlp_synth` directory with `train.jsonl`, `val.jsonl`, and `test.jsonl` files.

### 3. Training

To train the model, run the `pretrain_dlp.py` script with the `dlp_train` configuration:

```bash
python HRM/pretrain_dlp.py --config-name dlp_train
```

This will train the model using the configuration in `HRM/config/dlp_train.yaml` and save the checkpoints in the `checkpoints` directory.

### 4. Evaluation

To evaluate the trained model, run the `evaluate_dlp.py` script with the path to the checkpoint and the test data:

```bash
python HRM/evaluate_dlp.py --checkpoint <path_to_checkpoint> --data-path data/dlp_synth/test.jsonl
```

## Development Conventions

*   **Configuration:** The project uses [Hydra](https://hydra.cc/) for managing configurations. The main configuration file is `HRM/config/dlp_train.yaml`.
*   **Experiment Tracking:** The training script is integrated with [Weights & Biases](https://wandb.ai/) for experiment tracking.
*   **Code Style:** The code follows the PEP 8 style guide.
*   **Testing:** The project includes a `quick_start_dlp.py` script that can be used for end-to-end testing of the pipeline.
