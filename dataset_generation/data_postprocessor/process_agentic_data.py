"""
Main Post-Processing Script for Agentic DLP Data

Converts generated agentic data to training format with labels and spans.
"""

import json
import argparse
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm

from pii_extractor import PIIExtractor, PIISpan
from business_context_analyzer import BusinessContextAnalyzer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticDataProcessor:
    """Process agentic generated data into training format"""
    
    def __init__(self):
        """Initialize processors"""
        self.pii_extractor = PIIExtractor()
        self.context_analyzer = BusinessContextAnalyzer()
    
    def process_generated_data(
        self, 
        input_file: str, 
        output_file: str,
        preserve_metadata: bool = False
    ) -> Dict[str, int]:
        """
        Process generated JSONL data and convert to training format
        
        Args:
            input_file: Path to generated JSONL file
            output_file: Path to output training JSONL file  
            preserve_metadata: Keep generation metadata in output
            
        Returns:
            Processing statistics
        """
        
        logger.info(f"Processing {input_file} -> {output_file}")
        
        stats = {
            'total_examples': 0,
            'processed_examples': 0,
            'skipped_examples': 0,
            'total_spans': 0,
            'spans_by_type': {},
            'labels_by_type': {'sensitivity': 0, 'exposure': 0, 'context': 0, 'obfuscation': 0}
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, desc="Processing"), 1):
                try:
                    # Parse input line
                    raw_example = json.loads(line.strip())
                    stats['total_examples'] += 1
                    
                    # Convert to training format
                    training_example = self._convert_to_training_format(
                        raw_example, preserve_metadata
                    )
                    
                    if training_example:
                        # Write processed example
                        outfile.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                        stats['processed_examples'] += 1
                        
                        # Update statistics
                        self._update_stats(stats, training_example)
                    else:
                        stats['skipped_examples'] += 1
                        logger.warning(f"Skipped malformed example at line {line_num}")
                
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    stats['skipped_examples'] += 1
                    continue
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    def _convert_to_training_format(
        self, 
        raw_example: Dict[str, Any],
        preserve_metadata: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Convert raw generated example to training format"""
        
        try:
            # Extract core content
            channel = raw_example.get('channel', 'email')
            subject = raw_example.get('subject', '')
            body = raw_example.get('body', '')
            recipients = raw_example.get('recipients', [])
            attachments = raw_example.get('attachments', [])
            links = raw_example.get('links', [])
            
            # Combine text for PII extraction
            combined_text = f"{subject}\n\n{body}"
            if attachments:
                combined_text += f"\n\nAttachments: {', '.join(attachments)}"
            
            # Extract PII spans
            pii_spans = self.pii_extractor.extract_all_pii(combined_text)
            
            # Convert spans to expected format
            spans = []
            for span in pii_spans:
                spans.append({
                    'type': span.type,
                    'text': span.text,
                    'start': span.start,
                    'end': span.end,
                    'confidence': span.confidence
                })
            
            # Infer user role from metadata or content
            user_role = self._infer_user_role(raw_example)
            
            # Infer sender domain
            sender_domain = self.context_analyzer.infer_sender_domain(recipients, body)
            
            # Analyze business context
            context_analysis = self.context_analyzer.analyze_sharing_context(
                spans, user_role, recipients, body, sender_domain
            )
            
            # Detect obfuscation
            obfuscation_indicators = self.pii_extractor.detect_obfuscation_indicators(combined_text)
            has_obfuscation = any(obfuscation_indicators.values())
            
            # Generate document labels
            labels = {
                'sensitivity': self._calculate_sensitivity(spans, user_role),
                'exposure': context_analysis.exposure,
                'context': context_analysis.context,
                'obfuscation': 1 if has_obfuscation else 0
            }
            
            # Create training example
            training_example = {
                'channel': channel,
                'subject': subject,
                'body': body,
                'recipients': recipients,
                'attachments': attachments,
                'links': links,
                'labels': labels,
                'spans': spans,
                'user': {'role': user_role} if user_role else {},
                'thread': {},  # Empty for now, could be extended
                'meta': {}
            }
            
            # Add analysis metadata if requested
            if preserve_metadata:
                training_example['_analysis'] = {
                    'context_analysis': {
                        'risk_factors': context_analysis.risk_factors,
                        'legitimate_indicators': context_analysis.legitimate_indicators,
                        'recipient_analysis': context_analysis.recipient_analysis
                    },
                    'obfuscation_indicators': obfuscation_indicators,
                    'sender_domain': sender_domain,
                    'original_metadata': raw_example.get('_metadata', {})
                }
            
            return training_example
            
        except Exception as e:
            logger.error(f"Error converting example: {e}")
            return None
    
    def _infer_user_role(self, raw_example: Dict[str, Any]) -> Optional[str]:
        """Infer user role from metadata or content"""
        
        # Try metadata first
        metadata = raw_example.get('_metadata', {})
        if 'agent_type' in metadata:
            agent_type = metadata['agent_type']
            
            # Map agent types to user roles
            role_mapping = {
                'legal': 'LEGAL',
                'finance': 'FINANCE', 
                'hr': 'HR',
                'security': 'SECURITY',
                'clean_business': 'BUSINESS',
                'casual': 'EMPLOYEE'
            }
            
            return role_mapping.get(agent_type)
        
        # Try to infer from content
        body = raw_example.get('body', '').lower()
        subject = raw_example.get('subject', '').lower()
        combined = f"{subject} {body}"
        
        if any(term in combined for term in ['legal', 'attorney', 'counsel', 'law', 'nda']):
            return 'LEGAL'
        elif any(term in combined for term in ['hr', 'human resources', 'employee', 'personnel']):
            return 'HR'
        elif any(term in combined for term in ['finance', 'accounting', 'audit', 'payment']):
            return 'FINANCE'
        elif any(term in combined for term in ['security', 'incident', 'breach', 'vulnerability']):
            return 'SECURITY'
        else:
            return 'BUSINESS'  # Default
    
    def _calculate_sensitivity(self, spans: List[Dict], user_role: Optional[str]) -> int:
        """Calculate sensitivity label based on spans and context"""
        
        # High sensitivity PII types
        highly_sensitive = {'PAN', 'SSN', 'SECRET', 'DBURI'}
        moderately_sensitive = {'EMAIL', 'PHONE', 'NAME'}
        
        # Check for highly sensitive PII
        if any(span['type'] in highly_sensitive for span in spans):
            return 1
        
        # Multiple moderately sensitive spans
        moderate_count = sum(1 for span in spans if span['type'] in moderately_sensitive)
        if moderate_count >= 2:
            return 1
        
        # Role-specific sensitivity
        if user_role == 'LEGAL' and any(span['type'] in {'NDA', 'MATTER'} for span in spans):
            return 1
        
        # Default: not sensitive
        return 0
    
    def _update_stats(self, stats: Dict, example: Dict[str, Any]):
        """Update processing statistics"""
        
        # Count spans by type
        for span in example.get('spans', []):
            span_type = span['type']
            stats['spans_by_type'][span_type] = stats['spans_by_type'].get(span_type, 0) + 1
            stats['total_spans'] += 1
        
        # Count labels
        labels = example.get('labels', {})
        for label_type, value in labels.items():
            if value == 1:
                stats['labels_by_type'][label_type] += 1
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        file_pattern: str = "*_examples_augmented.jsonl",
        preserve_metadata: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """
        Process entire directory of generated data
        
        Args:
            input_dir: Input directory with generated JSONL files
            output_dir: Output directory for training files
            file_pattern: File pattern to match (default: augmented files)
            preserve_metadata: Keep generation metadata
            
        Returns:
            Processing statistics by file
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find files to process
        files_to_process = list(input_path.glob(file_pattern))
        if not files_to_process:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return {}
        
        all_stats = {}
        
        for input_file in files_to_process:
            # Determine output filename
            if 'train' in input_file.name:
                output_file = output_path / 'train.jsonl'
            elif 'val' in input_file.name:
                output_file = output_path / 'val.jsonl'
            elif 'test' in input_file.name:
                output_file = output_path / 'test.jsonl'
            else:
                # Use original name
                output_file = output_path / input_file.name.replace('_augmented', '_training')
            
            # Process file
            stats = self.process_generated_data(
                str(input_file), 
                str(output_file), 
                preserve_metadata
            )
            
            all_stats[str(input_file)] = stats
        
        # Write combined statistics
        stats_file = output_path / 'processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        logger.info(f"Processing complete. Stats saved to {stats_file}")
        return all_stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Process agentic DLP data for training")
    
    parser.add_argument('input', help="Input file or directory")
    parser.add_argument('output', help="Output file or directory")
    parser.add_argument('--preserve-metadata', action='store_true',
                       help="Preserve generation metadata in output")
    parser.add_argument('--file-pattern', default="*_examples_augmented.jsonl",
                       help="File pattern for directory processing")
    
    args = parser.parse_args()
    
    processor = AgenticDataProcessor()
    
    # Determine if processing single file or directory
    if os.path.isfile(args.input):
        # Single file processing
        stats = processor.process_generated_data(
            args.input, args.output, args.preserve_metadata
        )
        print(f"Processing complete: {stats}")
    
    elif os.path.isdir(args.input):
        # Directory processing
        all_stats = processor.process_directory(
            args.input, args.output, args.file_pattern, args.preserve_metadata
        )
        
        # Print summary
        total_processed = sum(stats['processed_examples'] for stats in all_stats.values())
        total_spans = sum(stats['total_spans'] for stats in all_stats.values())
        print(f"Directory processing complete: {total_processed} examples, {total_spans} spans")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())