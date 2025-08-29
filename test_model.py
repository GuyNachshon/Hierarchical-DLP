#!/usr/bin/env python3
"""
HRM-DLP Model Testing Suite

Tests the trained model on various scenarios to validate performance.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.dlp.hrm_model import HRMDLP, HRMDLPConfig
from src.dlp.tokenizer import DLPTokenizer, SimpleTokenizer, TokenizerConfig
from src.dlp.dataset import DLPDataset, DLPDatasetConfig
from src.dlp.dsl import DSLSerializer, ID_TO_BIO_TAG
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRMDLPTester:
    """Testing suite for trained HRM-DLP model."""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Default config matching training
            config_dict = {
                "vocab_size": 16000,
                "seq_len": 1024,
                "hidden_size": 384,
                "H_layers": 2,
                "L_layers": 8,
                "num_heads": 6,
                "expansion": 4.0,
                "num_doc_scores": 4,
                "memory_vec_dim": 256,
                "use_act": True,
                "act_max_steps": 4,
                "segment_size": 64,
                "forward_dtype": "bfloat16" if self.device.type == 'cuda' else "float32"
            }
        
        # Create model config
        self.model_config = HRMDLPConfig(**config_dict)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Create DSL serializer
        self.serializer = DSLSerializer()
        
        logger.info("HRM-DLP tester initialized successfully")
    
    def _load_model(self, checkpoint_path: str) -> HRMDLP:
        """Load trained model from checkpoint."""
        model = HRMDLP(self.model_config)
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, using random weights")
        
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")
        
        return model
    
    def _load_tokenizer(self):
        """Load the trained tokenizer."""
        tokenizer_path = "checkpoints/hrm_dlp/tokenizer.model"
        
        if os.path.exists(tokenizer_path):
            try:
                config = TokenizerConfig(vocab_size=self.model_config.vocab_size)
                tokenizer = DLPTokenizer(tokenizer_path, config)
                logger.info("Loaded SentencePiece tokenizer")
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece tokenizer: {e}")
        
        logger.info("Using SimpleTokenizer fallback")
        return SimpleTokenizer(vocab_size=self.model_config.vocab_size)
    
    def predict_single(self, example: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Run inference on a single example.
        
        Args:
            example: DLP example dictionary
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        with torch.no_grad():
            # Serialize to DSL format
            serialization_result = self.serializer.serialize(example)
            dsl_text = serialization_result.dsl_text
            
            if verbose:
                print(f"🔤 DSL Text (first 200 chars):")
                print(f"   {dsl_text[:200]}...")
                print()
            
            # Tokenize
            token_ids = self.tokenizer.encode(dsl_text)
            if len(token_ids) > self.model_config.seq_len:
                token_ids = token_ids[:self.model_config.seq_len]
            
            # Pad to max length
            attention_mask = [1] * len(token_ids)
            while len(token_ids) < self.model_config.seq_len:
                token_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)
            
            # Create batch
            batch = {
                'input_ids': torch.tensor([token_ids], dtype=torch.long).to(self.device),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long).to(self.device),
                'doc_labels': torch.zeros(1, 4).to(self.device),  # Dummy
                'bio_labels': torch.zeros(1, self.model_config.seq_len, dtype=torch.long).to(self.device),  # Dummy
                'memory_flags': torch.zeros(1, 256).to(self.device)
            }
            
            # Run inference with ACT loop
            carry = self.model.initial_carry(batch)
            total_steps = 0
            
            for step in range(self.model_config.act_max_steps):
                carry, output, all_halted = self.model(carry, batch)
                total_steps += 1
                if all_halted:
                    break
            
            # Extract predictions
            doc_probs = torch.sigmoid(output.doc_logits[0]).cpu().numpy()
            span_probs = torch.softmax(output.span_logits[0], dim=-1).cpu().numpy()
            memory_vector = output.memory_vector[0].cpu().numpy()
            
            # Get top BIO predictions for non-padding tokens
            valid_length = sum(attention_mask)
            span_predictions = []
            for i in range(valid_length):
                pred_id = np.argmax(span_probs[i])
                confidence = span_probs[i][pred_id]
                tag = ID_TO_BIO_TAG.get(pred_id, f"UNK_{pred_id}")
                if tag != 'O' or confidence > 0.8:  # Only show non-O tags or high confidence O
                    span_predictions.append({
                        'position': i,
                        'tag': tag,
                        'confidence': float(confidence)
                    })
            
            results = {
                'document_scores': {
                    'sensitivity': float(doc_probs[0]),
                    'exposure': float(doc_probs[1]),
                    'context': float(doc_probs[2]),
                    'obfuscation': float(doc_probs[3])
                },
                'span_predictions': span_predictions[:20],  # Top 20 non-O predictions
                'memory_vector_norm': float(np.linalg.norm(memory_vector)),
                'act_steps': total_steps,
                'total_tokens': valid_length,
                'decision_summary': self._make_decision(doc_probs)
            }
            
            if verbose:
                self._print_predictions(results, dsl_text)
            
            return results
    
    def _make_decision(self, doc_probs: np.ndarray) -> Dict[str, Any]:
        """Make DLP decision based on document scores."""
        sensitivity, exposure, context, obfuscation = doc_probs
        
        # Decision logic based on docs
        if sensitivity > 0.9 and exposure > 0.7:
            decision = "BLOCK"
            risk_level = "HIGH"
        elif sensitivity > 0.7 or exposure > 0.5:
            decision = "WARN"
            risk_level = "MEDIUM"
        elif sensitivity > 0.2 or exposure > 0.2:
            decision = "ALLOW_WITH_MONITORING"
            risk_level = "LOW"
        else:
            decision = "ALLOW"
            risk_level = "MINIMAL"
        
        return {
            'decision': decision,
            'risk_level': risk_level,
            'primary_concern': max([
                ('Sensitivity', sensitivity),
                ('Exposure Risk', exposure), 
                ('Context Issues', context),
                ('Obfuscation', obfuscation)
            ], key=lambda x: x[1])[0],
            'confidence': float(max(doc_probs))
        }
    
    def _print_predictions(self, results: Dict[str, Any], dsl_text: str):
        """Print formatted prediction results."""
        print("=" * 60)
        print("🤖 HRM-DLP ANALYSIS RESULTS")
        print("=" * 60)
        
        scores = results['document_scores']
        print(f"📊 Document Scores:")
        print(f"   Sensitivity:  {scores['sensitivity']:.3f}")
        print(f"   Exposure:     {scores['exposure']:.3f}")
        print(f"   Context:      {scores['context']:.3f}")
        print(f"   Obfuscation:  {scores['obfuscation']:.3f}")
        print()
        
        decision = results['decision_summary']
        print(f"⚖️  Decision: {decision['decision']} ({decision['risk_level']} risk)")
        print(f"🎯 Primary Concern: {decision['primary_concern']}")
        print(f"🎲 Confidence: {decision['confidence']:.3f}")
        print()
        
        if results['span_predictions']:
            print(f"🔖 Detected Spans (top {len(results['span_predictions'])}):")
            for span in results['span_predictions'][:10]:
                print(f"   {span['tag']:<15} @ pos {span['position']:<4} (conf: {span['confidence']:.3f})")
            print()
        
        print(f"⚡ ACT Steps: {results['act_steps']}")
        print(f"📏 Tokens: {results['total_tokens']}")
        print(f"🧠 Memory Vector Norm: {results['memory_vector_norm']:.2f}")
        print("=" * 60)
        print()
    
    def test_examples(self, examples: List[Dict[str, Any]], verbose: bool = True):
        """Test multiple examples and show aggregate results."""
        logger.info(f"Testing {len(examples)} examples...")
        
        all_results = []
        decision_counts = {}
        avg_scores = {'sensitivity': 0, 'exposure': 0, 'context': 0, 'obfuscation': 0}
        
        for i, example in enumerate(examples):
            if verbose:
                print(f"\n📧 Example {i+1}/{len(examples)}")
                print(f"Subject: {example.get('subject', 'N/A')}")
                print(f"Recipients: {example.get('recipients', [])}")
            
            try:
                result = self.predict_single(example, verbose=verbose and i < 3)  # Detailed for first 3
                all_results.append(result)
                
                decision = result['decision_summary']['decision']
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                
                for score_name, score_value in result['document_scores'].items():
                    avg_scores[score_name] += score_value
                
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}")
                continue
        
        # Compute averages
        if all_results:
            for score_name in avg_scores:
                avg_scores[score_name] /= len(all_results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 BATCH TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully processed: {len(all_results)}/{len(examples)} examples")
        print(f"📊 Average Document Scores:")
        for score_name, avg_score in avg_scores.items():
            print(f"   {score_name.capitalize():<12}: {avg_score:.3f}")
        
        print(f"\n⚖️  Decision Distribution:")
        for decision, count in decision_counts.items():
            pct = (count / len(all_results)) * 100 if all_results else 0
            print(f"   {decision:<20}: {count:>3} ({pct:5.1f}%)")
        
        avg_steps = np.mean([r['act_steps'] for r in all_results]) if all_results else 0
        print(f"\n⚡ Average ACT Steps: {avg_steps:.1f}")
        print("=" * 60)
        
        return all_results
    
    def test_from_dataset(self, jsonl_path: str, max_examples: int = 10):
        """Test examples from JSONL dataset file."""
        if not os.path.exists(jsonl_path):
            logger.error(f"Dataset file {jsonl_path} not found")
            return
        
        examples = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        return self.test_examples(examples)


def create_test_examples() -> List[Dict[str, Any]]:
    """Create synthetic test examples for different risk scenarios."""
    return [
        {
            "channel": "email",
            "user": {"role": "EMPLOYEE", "dept": "FINANCE"},
            "recipients": ["external@gmail.com"],
            "subject": "Q3 Financial Results - CONFIDENTIAL",
            "body": "Please find attached our Q3 financial statements. Revenue was $2.5M with EBITDA of $450K. SSN for verification: 123-45-6789. Please keep this confidential.",
            "attachments": [
                {"name": "financials_q3.xlsx", "size": 102400, "mime": "application/vnd.ms-excel"}
            ],
            "labels": {"sensitivity": 1, "exposure": 1, "context": 1, "obfuscation": 0}
        },
        {
            "channel": "email", 
            "user": {"role": "INTERN", "dept": "HR"},
            "recipients": ["colleague@company.com"],
            "subject": "Meeting notes",
            "body": "Here are the notes from today's team meeting. We discussed project timelines and resource allocation.",
            "attachments": [],
            "labels": {"sensitivity": 0, "exposure": 0, "context": 1, "obfuscation": 0}
        },
        {
            "channel": "email",
            "user": {"role": "LEGAL", "dept": "LEGAL"},
            "recipients": ["opposing-counsel@lawfirm.com"],
            "subject": "Re: Settlement Negotiations",
            "body": "Our client is prepared to offer $500,000 as final settlement. Database connection string: postgres://user:p@ssw0rd@db.company.com:5432/sensitive",
            "attachments": [
                {"name": "settlement_terms.pdf", "size": 51200, "mime": "application/pdf"}
            ],
            "labels": {"sensitivity": 1, "exposure": 0, "context": 1, "obfuscation": 1}
        }
    ]


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained HRM-DLP model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hrm_dlp/checkpoint_latest.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config JSON")
    parser.add_argument("--dataset", type=str, help="Path to JSONL test dataset")
    parser.add_argument("--max_examples", type=int, default=10, help="Max examples to test from dataset")
    parser.add_argument("--synthetic", action="store_true", help="Test with synthetic examples")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = HRMDLPTester(args.checkpoint, args.config)
    
    if args.dataset:
        print(f"🧪 Testing on dataset: {args.dataset}")
        results = tester.test_from_dataset(args.dataset, args.max_examples)
    elif args.synthetic:
        print("🧪 Testing on synthetic examples")
        test_examples = create_test_examples()
        results = tester.test_examples(test_examples)
    else:
        print("🧪 Testing single synthetic example")
        test_example = create_test_examples()[0]
        result = tester.predict_single(test_example)
        print("✅ Single test completed!")


if __name__ == "__main__":
    main()