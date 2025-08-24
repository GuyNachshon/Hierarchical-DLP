"""
Evaluation Utilities

Common functions for model evaluation and metric computation.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_hrm_metrics(predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
    """
    Compute metrics for HRM puzzle solving tasks
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
    
    Returns:
        Dictionary of computed metrics
    """
    # Exact match accuracy (primary metric for HRM)
    exact_matches = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    exact_accuracy = exact_matches / len(predictions) if predictions else 0.0
    
    return {
        "exact_accuracy": exact_accuracy,
        "total_examples": len(predictions),
        "correct_predictions": exact_matches
    }


def compute_dlp_metrics(doc_predictions: np.ndarray, doc_targets: np.ndarray,
                       span_predictions: np.ndarray, span_targets: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for DLP tasks
    
    Args:
        doc_predictions: Document-level predictions (N,)
        doc_targets: Document-level targets (N,)
        span_predictions: Token-level span predictions (N, seq_len)
        span_targets: Token-level span targets (N, seq_len)
    
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Document-level metrics
    doc_accuracy = accuracy_score(doc_targets, doc_predictions)
    metrics["document_accuracy"] = doc_accuracy
    
    # Token-level metrics (flatten sequences)
    span_pred_flat = span_predictions.flatten()
    span_target_flat = span_targets.flatten()
    
    # Filter out padding tokens (assuming -100 is padding)
    mask = span_target_flat != -100
    if mask.sum() > 0:
        span_pred_masked = span_pred_flat[mask]
        span_target_masked = span_target_flat[mask]
        
        # Compute token-level metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            span_target_masked, span_pred_masked, average="weighted", zero_division=0
        )
        
        metrics.update({
            "token_precision": precision,
            "token_recall": recall, 
            "token_f1": f1,
            "token_accuracy": accuracy_score(span_target_masked, span_pred_masked)
        })
        
        # PII-specific metrics (non-zero classes)
        pii_mask = span_target_masked > 0
        if pii_mask.sum() > 0:
            pii_precision, pii_recall, pii_f1, _ = precision_recall_fscore_support(
                span_target_masked[pii_mask], span_pred_masked[pii_mask], 
                average="weighted", zero_division=0
            )
            metrics.update({
                "pii_precision": pii_precision,
                "pii_recall": pii_recall,
                "pii_f1": pii_f1
            })
    
    return metrics


def compute_metrics(predictions: Dict[str, Any], targets: Dict[str, Any], 
                   task_type: str) -> Dict[str, float]:
    """
    Unified metric computation interface
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets  
        task_type: Type of task ("hrm" or "dlp")
    
    Returns:
        Dictionary of computed metrics
    """
    if task_type == "hrm":
        return compute_hrm_metrics(predictions["outputs"], targets["outputs"])
    elif task_type == "dlp":
        return compute_dlp_metrics(
            predictions["doc_labels"], targets["doc_labels"],
            predictions["span_labels"], targets["span_labels"] 
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                  device: str, task_type: str) -> Dict[str, float]:
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        task_type: Type of task ("hrm" or "dlp")
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move batch to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != "targets"}
        targets = batch["targets"]
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute loss
        if hasattr(outputs, "loss"):
            total_loss += outputs.loss.item()
        
        # Collect predictions and targets
        if task_type == "hrm":
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())
        elif task_type == "dlp":
            # Handle multi-task DLP outputs
            doc_pred = outputs.doc_logits.argmax(dim=-1).cpu().numpy()
            span_pred = outputs.span_logits.argmax(dim=-1).cpu().numpy()
            
            all_predictions.append({
                "doc_labels": doc_pred,
                "span_labels": span_pred
            })
            all_targets.append({
                "doc_labels": targets["doc_labels"].cpu().numpy(),
                "span_labels": targets["span_labels"].cpu().numpy()
            })
        
        num_batches += 1
    
    # Compute metrics
    if task_type == "dlp":
        # Concatenate DLP predictions and targets
        doc_predictions = np.concatenate([p["doc_labels"] for p in all_predictions])
        doc_targets = np.concatenate([t["doc_labels"] for t in all_targets])
        span_predictions = np.concatenate([p["span_labels"] for p in all_predictions])
        span_targets = np.concatenate([t["span_labels"] for t in all_targets])
        
        metrics = compute_dlp_metrics(doc_predictions, doc_targets, span_predictions, span_targets)
    else:
        metrics = compute_hrm_metrics(all_predictions, all_targets)
    
    # Add loss to metrics
    if num_batches > 0:
        metrics["eval_loss"] = total_loss / num_batches
    
    return metrics