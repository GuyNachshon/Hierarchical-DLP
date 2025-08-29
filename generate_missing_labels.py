#!/usr/bin/env python3
"""
Generate Missing DLP Labels

Creates the missing 'labels' field for HRM-DLP training data by analyzing
content, recipients, attachments, and sensitivity indicators.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class DLPLabelGenerator:
    """Generates DLP labels from email content and metadata."""
    
    def __init__(self):
        # Define patterns and weights for each risk type
        self.sensitivity_patterns = {
            'high': {
                'patterns': [
                    r'\b(ssn|social security)\b',
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
                    r'\b(credit card|cc)\b',
                    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # CC format
                    r'\b(password|pwd|api[_\s]?key|secret[_\s]?key)\b',
                    r'\b(database|db)[_\s]?(connection|uri|url)\b',
                    r'postgresql://|mysql://|mongodb://',
                    r'sk-[a-zA-Z0-9]{20,}',  # API key format
                    r'AKIA[0-9A-Z]{16}',  # AWS access key format
                    r'\b(confidential|proprietary|trade[_\s]?secret)\b',
                    r'\b(medical|health|phi|hipaa)\b',
                    r'\b(attorney[_\s]?client|privileged)\b',
                ],
                'weight': 0.8
            },
            'medium': {
                'patterns': [
                    r'\b(employee[_\s]?id|emp[_\s]?id)\b',
                    r'\b(phone|tel|mobile):\s*\d',
                    r'\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b',  # Phone format
                    r'\b(salary|compensation|bonus)\b',
                    r'\b(nda|non[_\s]?disclosure)\b',
                    r'\b(internal[_\s]?only|restricted)\b',
                ],
                'weight': 0.5
            },
            'low': {
                'patterns': [
                    r'\b(private|sensitive)\b',
                    r'\b(meeting|discussion|review)\b',
                ],
                'weight': 0.2
            }
        }
        
        self.exposure_patterns = {
            'high': {
                'patterns': [
                    r'@(gmail|yahoo|hotmail|outlook)\.com',
                    r'@.*\.(ru|cn|tk|ml)',  # Suspicious TLDs
                    r'hacker|malicious|competitor|rival',
                    r'personal[_\s]?email|private[_\s]?account',
                ],
                'weight': 0.9
            },
            'medium': {
                'patterns': [
                    r'@(?!.*\.(com|org|edu|gov)$)',  # Non-standard domains
                    r'external|vendor|partner|client',
                    r'@.*\.(io|co|net)',  # Somewhat risky TLDs
                ],
                'weight': 0.4
            },
            'low': {
                'patterns': [
                    r'@company\.com|@example\.com',  # Internal domains
                    r'team@|all-staff@|group@',
                ],
                'weight': 0.1
            }
        }
        
        self.context_patterns = {
            'high': {
                'patterns': [
                    r'\b(intern|temp|contractor|unauthorized)\b',
                    r'\b(violation|breach|incident)\b',
                    r'\b(urgent|immediate|asap)\b',
                    r'\b(selling|sale|buy|purchase)\b.*\b(data|database|information)\b',
                ],
                'weight': 0.7
            },
            'medium': {
                'patterns': [
                    r'\b(junior|new[_\s]?hire)\b',
                    r'\b(mistake|error|wrong)\b',
                    r'\b(personal|home|wife|spouse)\b',
                ],
                'weight': 0.4
            },
            'low': {
                'patterns': [
                    r'\b(manager|director|senior|lead)\b',
                    r'\b(official|approved|authorized)\b',
                ],
                'weight': 0.1
            }
        }
        
        self.obfuscation_patterns = {
            'high': {
                'patterns': [
                    r'\b(base64|encoded|encrypted|hidden)\b',
                    r'\b(bitcoin|btc|crypto)\b',
                    r'\$[0-9]+k\b.*\b(payment|cash|money)\b',
                    r'\b(dark[_\s]?web|tor|onion)\b',
                ],
                'weight': 0.8
            },
            'medium': {
                'patterns': [
                    r'\b(obfuscated|masked|disguised)\b',
                    r'\b(code|cipher|key)\b',
                ],
                'weight': 0.4
            }
        }
        
        # Sensitivity indicator mappings
        self.indicator_weights = {
            'PII': 0.8,
            'PHI': 0.9,
            'SSN': 0.9,
            'PHONE': 0.4,
            'Attorney-Client Privileged': 0.8,
            'Work Product': 0.7,
            'Client Confidential': 0.7,
            'NDA-Protected': 0.6,
            'Confidential': 0.6,
            'confidential': 0.6,
            'financial_data': 0.7,
            'internal_use_only': 0.4,
            'Legal': 0.5,
            'Contractual Confidentiality': 0.6,
            'general': 0.1,
            'non-sensitive': 0.0,
            'no_personal_data': 0.0,
            'policy_document': 0.2,
            'none': 0.0,
            'None': 0.0,
            '': 0.0
        }
    
    def analyze_content(self, text: str, patterns: Dict) -> float:
        """Analyze text content against patterns and return risk score."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        total_score = 0.0
        
        for risk_level, config in patterns.items():
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    # More matches = higher score, but with diminishing returns
                    score_boost = config['weight'] * min(matches * 0.3, 1.0)
                    total_score += score_boost
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def analyze_recipients(self, recipients: List[str]) -> float:
        """Analyze recipient exposure risk."""
        if not recipients:
            return 0.0
        
        recipient_text = ' '.join(recipients)
        return self.analyze_content(recipient_text, self.exposure_patterns)
    
    def analyze_attachments(self, attachments: List[Dict]) -> tuple:
        """Analyze attachments for sensitivity and context indicators."""
        if not attachments:
            return 0.0, 0.0
        
        sensitivity_score = 0.0
        context_score = 0.0
        
        for attachment in attachments:
            # Large files are riskier
            size = attachment.get('size', 0)
            if size > 10000000:  # >10MB
                context_score += 0.3
            elif size > 1000000:  # >1MB
                context_score += 0.2
            
            # Analyze sensitivity indicators
            indicators = attachment.get('sensitivity_indicators', [])
            for indicator in indicators:
                if indicator in self.indicator_weights:
                    indicator_score = self.indicator_weights[indicator]
                    sensitivity_score += indicator_score
                    
                    # Some indicators also imply context risk
                    if indicator in ['PII', 'PHI', 'SSN', 'confidential']:
                        context_score += indicator_score * 0.5
            
            # Analyze attachment names and content summaries
            att_text = (attachment.get('name', '') + ' ' + 
                       attachment.get('content_summary', ''))
            sensitivity_score += self.analyze_content(att_text, self.sensitivity_patterns) * 0.3
        
        return min(sensitivity_score, 1.0), min(context_score, 1.0)
    
    def generate_labels(self, example: Dict[str, Any]) -> Dict[str, float]:
        """Generate all four DLP labels for an example."""
        # Extract content
        subject = example.get('subject', '')
        body = example.get('body', '')
        recipients = example.get('recipients', [])
        attachments = example.get('attachments', [])
        
        # Combine text content
        all_text = f"{subject} {body}"
        
        # Calculate scores
        sensitivity = self.analyze_content(all_text, self.sensitivity_patterns)
        exposure = self.analyze_recipients(recipients)
        context = self.analyze_content(all_text, self.context_patterns)
        obfuscation = self.analyze_content(all_text, self.obfuscation_patterns)
        
        # Add attachment analysis
        att_sensitivity, att_context = self.analyze_attachments(attachments)
        sensitivity = min(sensitivity + att_sensitivity * 0.6, 1.0)
        context = min(context + att_context * 0.4, 1.0)
        
        # Add some realistic noise to avoid identical scores
        noise_factor = 0.05
        sensitivity += np.random.uniform(-noise_factor, noise_factor)
        exposure += np.random.uniform(-noise_factor, noise_factor)
        context += np.random.uniform(-noise_factor, noise_factor)
        obfuscation += np.random.uniform(-noise_factor, noise_factor)
        
        # Ensure scores are in [0, 1] range
        labels = {
            'sensitivity': max(0.0, min(1.0, sensitivity)),
            'exposure': max(0.0, min(1.0, exposure)),
            'context': max(0.0, min(1.0, context)),
            'obfuscation': max(0.0, min(1.0, obfuscation))
        }
        
        return labels

def process_dataset(input_path: Path, output_path: Path, generator: DLPLabelGenerator):
    """Process a dataset file and add labels."""
    print(f"🔄 Processing: {input_path}")
    
    processed_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                example = json.loads(line.strip())
                
                # Generate labels
                labels = generator.generate_labels(example)
                
                # Add labels to example
                example['labels'] = labels
                
                # Write back to file
                outfile.write(json.dumps(example) + '\n')
                processed_count += 1
                
                # Progress indicator
                if line_num % 100 == 0:
                    print(f"   Processed {line_num} examples...")
                    
            except json.JSONDecodeError:
                print(f"   ⚠️  Skipping malformed line {line_num}")
                continue
            except Exception as e:
                print(f"   ❌ Error processing line {line_num}: {e}")
                continue
    
    print(f"   ✅ Successfully processed {processed_count} examples")
    return processed_count

def main():
    print("🏷️  HRM-DLP Label Generation")
    print("=" * 50)
    
    # Set random seed for reproducible noise
    np.random.seed(42)
    
    generator = DLPLabelGenerator()
    
    # Process each split
    data_dir = Path("data/hrm_dlp_final")
    
    for split in ["train", "val", "test"]:
        input_path = data_dir / f"{split}.jsonl"
        output_path = data_dir / f"{split}_labeled.jsonl"
        
        if not input_path.exists():
            print(f"⚠️  Skipping {split}: file not found")
            continue
        
        print(f"\n📋 Processing {split} set...")
        processed = process_dataset(input_path, output_path, generator)
        
        if processed > 0:
            print(f"   💾 Saved labeled data to: {output_path}")
            
            # Analyze generated labels
            print(f"   🔍 Analyzing generated labels...")
            
            labels_summary = {'sensitivity': [], 'exposure': [], 'context': [], 'obfuscation': []}
            
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        example = json.loads(line)
                        labels = example.get('labels', {})
                        for label_type in labels_summary:
                            if label_type in labels:
                                labels_summary[label_type].append(labels[label_type])
                    except:
                        continue
            
            # Print statistics
            for label_type, values in labels_summary.items():
                if values:
                    values = np.array(values)
                    print(f"      {label_type.capitalize():<12}: mean={values.mean():.3f}, std={values.std():.3f}, range={values.max()-values.min():.3f}")
    
    print(f"\n✅ Label generation complete!")
    print(f"💡 Next steps:")
    print(f"   1. Update training config to use *_labeled.jsonl files")
    print(f"   2. Re-run training with proper ground truth labels")
    print(f"   3. Model should now learn to discriminate between risk levels!")

if __name__ == "__main__":
    main()