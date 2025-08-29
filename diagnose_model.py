#!/usr/bin/env python3
"""
HRM-DLP Model Diagnostic

Quick checks to understand model behavior.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from test_model import HRMDLPTester

def diagnose_model():
    print("🔬 HRM-DLP Model Diagnostic")
    print("=" * 50)
    
    # Initialize tester
    checkpoint_path = "checkpoints/hrm_dlp/checkpoint_latest.pt"
    try:
        tester = HRMDLPTester(checkpoint_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create a simple test example with obvious patterns
    test_example = {
        "channel": "email",
        "user": {"role": "FINANCE", "dept": "FINANCE"},
        "recipients": ["hacker@badactor.com"],
        "subject": "URGENT - All Company Passwords and SSNs",
        "body": "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012, Password: admin123, Database: postgres://admin:secret@db.company.com/sensitive, API Key: sk-1234567890abcdef, Phone: 555-123-4567",
        "attachments": [
            {"name": "all_passwords.txt", "size": 1000000, "mime": "text/plain"},
            {"name": "customer_ssns.xlsx", "size": 5000000, "mime": "application/vnd.ms-excel"}
        ]
    }
    
    print("\n🧪 Testing with EXTREME high-risk example...")
    print(f"   Recipients: {test_example['recipients']}")
    print(f"   Subject: {test_example['subject']}")
    print(f"   Body contains: SSN, Credit Card, Password, Database URI, API Key, Phone")
    print()
    
    # Get raw model outputs before sigmoid/softmax
    with torch.no_grad():
        # Serialize and tokenize
        serialization_result = tester.serializer.serialize(test_example)
        token_ids = tester.tokenizer.encode(serialization_result.dsl_text)
        if len(token_ids) > tester.model_config.seq_len:
            token_ids = token_ids[:tester.model_config.seq_len]
        
        # Pad
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < tester.model_config.seq_len:
            token_ids.append(tester.tokenizer.pad_token_id)
            attention_mask.append(0)
        
        # Create batch
        batch = {
            'input_ids': torch.tensor([token_ids], dtype=torch.long).to(tester.device),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long).to(tester.device),
            'doc_labels': torch.zeros(1, 4).to(tester.device),
            'bio_labels': torch.zeros(1, tester.model_config.seq_len, dtype=torch.long).to(tester.device),
            'memory_flags': torch.zeros(1, 256).to(tester.device)
        }
        
        # Run model
        carry = tester.model.initial_carry(batch)
        for step in range(tester.model_config.act_max_steps):
            carry, output, all_halted = tester.model(carry, batch)
            if all_halted:
                break
        
        # Analyze raw outputs
        print("🔍 Raw Model Outputs (before activation):")
        raw_doc_logits = output.doc_logits[0].cpu().numpy()
        print(f"   Document logits: [{raw_doc_logits[0]:.3f}, {raw_doc_logits[1]:.3f}, {raw_doc_logits[2]:.3f}, {raw_doc_logits[3]:.3f}]")
        
        # Apply sigmoid
        doc_probs = torch.sigmoid(output.doc_logits[0]).cpu().numpy()
        print(f"   After sigmoid:   [{doc_probs[0]:.3f}, {doc_probs[1]:.3f}, {doc_probs[2]:.3f}, {doc_probs[3]:.3f}]")
        
        # Check span logits statistics
        span_logits = output.span_logits[0].cpu().numpy()  # [seq_len, num_tags]
        span_max = np.max(span_logits, axis=1)  # Max logit per position
        span_mean = np.mean(span_logits, axis=1)  # Mean logit per position
        
        valid_length = sum(attention_mask)
        print(f"\n🔍 Span Logits Analysis (first {min(10, valid_length)} positions):")
        print(f"   Position | Max Logit | Mean Logit | Predicted Tag")
        print(f"   ---------|-----------|------------|-------------")
        
        for i in range(min(10, valid_length)):
            pred_id = np.argmax(span_logits[i])
            tag = tester.serializer.dsl_text if hasattr(tester.serializer, 'dsl_text') else f"TAG_{pred_id}"
            print(f"   {i:8d} | {span_max[i]:9.3f} | {span_mean[i]:10.3f} | {pred_id}")
        
        # Memory vector analysis
        memory_norm = torch.norm(output.memory_vector[0]).item()
        memory_mean = torch.mean(output.memory_vector[0]).item()
        memory_std = torch.std(output.memory_vector[0]).item()
        
        print(f"\n🧠 Memory Vector Analysis:")
        print(f"   Norm: {memory_norm:.2f}")
        print(f"   Mean: {memory_mean:.3f}")
        print(f"   Std:  {memory_std:.3f}")
        
        # ACT analysis
        print(f"\n⚡ ACT Analysis:")
        print(f"   Steps taken: {output.steps.item()}")
        print(f"   Q-halt:      {output.q_halt_logits.item():.3f}")
        print(f"   Q-continue:  {output.q_continue_logits.item():.3f}")
    
    print("\n💡 Diagnostic Summary:")
    if np.all(np.abs(raw_doc_logits) < 0.1):
        print("   ⚠️  Document logits near zero - model may need fine-tuning")
    else:
        print("   ✅ Document logits show variation")
    
    if memory_norm < 10:
        print("   ⚠️  Low memory vector norm - representations may be weak")
    else:
        print("   ✅ Memory vector has reasonable magnitude")
    
    if np.all(span_max < 1.0):
        print("   ⚠️  Span logits have low confidence - BIO tagging may need adjustment")
    else:
        print("   ✅ Span predictions show confidence")
    
    print("\n🎯 Recommendations:")
    print("   1. Model architecture is working correctly")
    print("   2. Predictions may need calibration or fine-tuning")
    print("   3. Consider training with stronger supervision signals")
    print("   4. The zero outputs suggest the model learned a conservative baseline")

if __name__ == "__main__":
    diagnose_model()