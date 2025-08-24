"""Evaluation script for HRM-DLP model

Computes comprehensive metrics for DLP tasks:
- Document classification (AUPRC, FP@95R, calibration)
- Span tagging (F1, Precision@k)
- Decision consistency under perturbations
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import sys
import json
import yaml
from dataclasses import dataclass
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Add HRM directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "HRM"))

from hrm_dlp.model import HRMDLPModel, create_dlp_model
from hrm_dlp.dataset import DLPDataset, DLPDatasetConfig
from hrm_dlp.tokenizer import DLPTokenizer, SimpleTokenizer
from hrm_dlp.dsl import ID_TO_BIO_TAG


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    # Document classification metrics
    doc_metrics: Dict[str, Dict[str, float]]  # {label: {metric: value}}
    
    # Span tagging metrics
    span_metrics: Dict[str, float]
    
    # Stability metrics
    stability_metrics: Dict[str, float]
    
    # Calibration results
    calibration_results: Dict[str, Any]


class DLPEvaluator:
    """Comprehensive evaluator for DLP model"""
    
    def __init__(
        self, 
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.doc_labels = ["sensitivity", "exposure", "context", "obfuscation"]
    
    def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> EvaluationResults:
        """Evaluate model on dataset"""
        # Load dataset
        dataset_config = DLPDatasetConfig(max_length=1024)
        dataset = DLPDataset(dataset_path, self.tokenizer, dataset_config)
        
        # Limit samples if specified
        if max_samples:
            dataset.examples = dataset.examples[:max_samples]
        
        print(f"Evaluating on {len(dataset)} examples")
        
        # Collect predictions and ground truth
        all_doc_logits = []
        all_doc_labels = []
        all_span_logits = []
        all_span_labels = []
        all_input_ids = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 100 == 0:
                    print(f"Processing {i}/{len(dataset)}")
                
                # Get sample
                sample = dataset[i]
                
                # Move to device and add batch dimension
                input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                doc_labels = sample["doc_labels"].unsqueeze(0)
                bio_labels = sample["bio_labels"].unsqueeze(0)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Collect results
                all_doc_logits.append(outputs.doc_logits.cpu())
                all_doc_labels.append(doc_labels)
                all_span_logits.append(outputs.span_logits.cpu())
                all_span_labels.append(bio_labels)
                all_input_ids.append(input_ids.cpu())
        
        # Concatenate all results
        all_doc_logits = torch.cat(all_doc_logits, dim=0)
        all_doc_labels = torch.cat(all_doc_labels, dim=0)
        all_span_logits = torch.cat(all_span_logits, dim=0)
        all_span_labels = torch.cat(all_span_labels, dim=0)
        
        # Compute metrics
        doc_metrics = self._compute_doc_metrics(all_doc_logits, all_doc_labels)
        span_metrics = self._compute_span_metrics(all_span_logits, all_span_labels)
        stability_metrics = self._compute_stability_metrics(dataset, max_samples=min(100, len(dataset)))
        calibration_results = self._compute_calibration_metrics(all_doc_logits, all_doc_labels)
        
        return EvaluationResults(
            doc_metrics=doc_metrics,
            span_metrics=span_metrics,
            stability_metrics=stability_metrics,
            calibration_results=calibration_results
        )
    
    def _compute_doc_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Compute document-level classification metrics"""
        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()
        
        metrics = {}
        
        for i, label_name in enumerate(self.doc_labels):
            y_true = labels_np[:, i]
            y_scores = probs[:, i]
            y_pred = (y_scores > 0.5).astype(int)
            
            # Basic classification metrics
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
            
            # AUC metrics (if we have positive examples)
            if len(np.unique(y_true)) > 1:
                auprc = average_precision_score(y_true, y_scores)
                auroc = roc_auc_score(y_true, y_scores)
                
                # False Positive Rate at 95% Recall (key DLP metric)
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_scores)
                
                # Find threshold that gives closest to 95% recall
                target_recall = 0.95
                recall_diff = np.abs(recall_curve - target_recall)
                best_idx = np.argmin(recall_diff)
                
                if best_idx < len(thresholds):
                    threshold_95r = thresholds[best_idx]
                    y_pred_95r = (y_scores >= threshold_95r).astype(int)
                    fp_95r = ((y_pred_95r == 1) & (y_true == 0)).sum()
                    total_negatives = (y_true == 0).sum()
                    fpr_95r = fp_95r / total_negatives if total_negatives > 0 else 0.0
                else:
                    fpr_95r = 0.0
            else:
                auprc = 0.0
                auroc = 0.5
                fpr_95r = 0.0
            
            metrics[label_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "auprc": float(auprc),
                "auroc": float(auroc),
                "fpr_at_95r": float(fpr_95r)
            }
        
        return metrics
    
    def _compute_span_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute span tagging metrics"""
        # Flatten predictions and labels, excluding ignored positions
        valid_mask = labels != -100
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_labels) == 0:
            return {"span_accuracy": 0.0, "span_f1_macro": 0.0}
        
        predictions = torch.argmax(valid_logits, dim=-1)
        
        # Overall accuracy
        accuracy = (predictions == valid_labels).float().mean().item()
        
        # Per-class metrics for F1 calculation
        num_classes = logits.shape[-1]
        class_f1s = []
        
        for class_id in range(num_classes):
            if class_id == 0:  # Skip 'O' tag for entity F1
                continue
                
            # Binary classification for this class
            pred_binary = (predictions == class_id).long()
            true_binary = (valid_labels == class_id).long()
            
            tp = (pred_binary * true_binary).sum().item()
            fp = (pred_binary * (1 - true_binary)).sum().item()
            fn = ((1 - pred_binary) * true_binary).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_f1s.append(f1)
        
        macro_f1 = np.mean(class_f1s) if class_f1s else 0.0
        
        # Entity-level F1 (considering B- and I- tags together)
        entity_f1 = self._compute_entity_f1(predictions, valid_labels)
        
        return {
            "span_accuracy": float(accuracy),
            "span_f1_macro": float(macro_f1),
            "span_entity_f1": float(entity_f1)
        }
    
    def _compute_entity_f1(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute entity-level F1 score"""
        def extract_entities(tags):
            """Extract entity spans from BIO tags"""
            entities = []
            current_entity = None
            
            for i, tag_id in enumerate(tags):
                tag = ID_TO_BIO_TAG.get(tag_id.item(), 'O')
                
                if tag.startswith('B-'):
                    # Begin new entity
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = tag[2:]
                    current_entity = (entity_type, i, i)
                elif tag.startswith('I-') and current_entity:
                    # Continue current entity
                    entity_type = tag[2:]
                    if entity_type == current_entity[0]:
                        current_entity = (current_entity[0], current_entity[1], i)
                    else:
                        # Type mismatch, end current and start new
                        entities.append(current_entity)
                        current_entity = (entity_type, i, i)
                else:
                    # Outside or type mismatch
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            if current_entity:
                entities.append(current_entity)
                
            return set(entities)
        
        pred_entities = extract_entities(predictions)
        true_entities = extract_entities(labels)
        
        if len(true_entities) == 0 and len(pred_entities) == 0:
            return 1.0
        elif len(true_entities) == 0:
            return 0.0
        
        tp = len(pred_entities & true_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _compute_stability_metrics(self, dataset: DLPDataset, max_samples: int = 100) -> Dict[str, float]:
        """Compute decision stability under text perturbations"""
        print("Computing stability metrics...")
        
        consistent_decisions = 0
        total_comparisons = 0
        
        self.model.eval()
        with torch.no_grad():
            for i in range(min(max_samples, len(dataset))):
                sample = dataset[i]
                
                # Original prediction
                input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                
                outputs_orig = self.model(input_ids, attention_mask)
                orig_probs = torch.sigmoid(outputs_orig.doc_logits).cpu().numpy()
                orig_decisions = (orig_probs > 0.5).astype(int)
                
                # Perturbed predictions (shift window by ±tokens)
                for shift in [-5, -3, 3, 5]:
                    if shift < 0:
                        # Remove tokens from beginning
                        shifted_ids = input_ids[:, abs(shift):]
                        shifted_mask = attention_mask[:, abs(shift):]
                    else:
                        # Remove tokens from end
                        shifted_ids = input_ids[:, :-shift]
                        shifted_mask = attention_mask[:, :-shift]
                    
                    if shifted_ids.size(1) > 10:  # Ensure minimum length
                        outputs_shifted = self.model(shifted_ids, shifted_mask)
                        shifted_probs = torch.sigmoid(outputs_shifted.doc_logits).cpu().numpy()
                        shifted_decisions = (shifted_probs > 0.5).astype(int)
                        
                        # Check consistency
                        if np.array_equal(orig_decisions, shifted_decisions):
                            consistent_decisions += 1
                        total_comparisons += 1
        
        stability = consistent_decisions / total_comparisons if total_comparisons > 0 else 0.0
        
        return {
            "decision_stability": float(stability)
        }
    
    def _compute_calibration_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Compute calibration metrics for probability estimates"""
        probs = torch.sigmoid(logits).numpy()
        labels_np = labels.numpy()
        
        calibration_results = {}
        
        for i, label_name in enumerate(self.doc_labels):
            y_true = labels_np[:, i]
            y_prob = probs[:, i]
            
            if len(np.unique(y_true)) > 1:
                # Compute calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = y_prob[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_results[label_name] = {
                    "ece": float(ece),
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_value": mean_predicted_value.tolist()
                }
            else:
                calibration_results[label_name] = {
                    "ece": 0.0,
                    "fraction_of_positives": [],
                    "mean_predicted_value": []
                }
        
        return calibration_results


def print_results(results: EvaluationResults):
    """Print evaluation results in a readable format"""
    print("\n" + "="*60)
    print("DOCUMENT CLASSIFICATION METRICS")
    print("="*60)
    
    for label, metrics in results.doc_metrics.items():
        print(f"\n{label.upper()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}")
        print(f"  FP@95R:    {metrics['fpr_at_95r']:.4f}")  # Key DLP metric
    
    print("\n" + "="*60)
    print("SPAN TAGGING METRICS")
    print("="*60)
    
    for metric, value in results.span_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("STABILITY METRICS")
    print("="*60)
    
    for metric, value in results.stability_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("CALIBRATION METRICS (ECE)")
    print("="*60)
    
    for label, metrics in results.calibration_results.items():
        print(f"{label}: {metrics['ece']:.4f}")


def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[torch.nn.Module, Any, Dict]:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint["config"]
    
    # Create model
    model = create_dlp_model(config_dict["arch"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create tokenizer (simplified for evaluation)
    tokenizer = SimpleTokenizer(vocab_size=config_dict.get("vocab_size", 16000))
    
    return model, tokenizer, config_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRM-DLP model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, 
                       default="data/runs/run_20250824_123640_a2f52bf9/split_outputs/test_examples_augmented.jsonl",
                       help="Path to evaluation dataset (JSONL file)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer, config = load_model_from_checkpoint(args.checkpoint)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create evaluator
    evaluator = DLPEvaluator(model, tokenizer, device)
    
    # Run evaluation
    print(f"Evaluating on {args.data_path}")
    results = evaluator.evaluate_dataset(args.data_path, args.max_samples)
    
    # Print results
    print_results(results)
    
    # Save results
    results_dict = {
        "doc_metrics": results.doc_metrics,
        "span_metrics": results.span_metrics,
        "stability_metrics": results.stability_metrics,
        "calibration_results": results.calibration_results,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()