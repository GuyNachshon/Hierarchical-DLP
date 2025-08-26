"""
Format converters for DLP training data.
"""

import random
from typing import Dict, List, Any, Optional
from utils.patterns import SpanDetector
from utils.helpers import generate_timestamp, hash_string


class DLPFormatConverter:
    """Converts generated examples to DLP training format."""
    
    def __init__(self):
        self.span_detector = SpanDetector()
    
    def convert_to_dlp_format(self, example: Any) -> Optional[Dict]:
        """Convert GeneratedExample to DLP training format."""
        try:
            # Extract spans from body
            body = getattr(example, 'body', '')
            spans = self.span_detector.extract_all_spans(body)
            
            # Generate document labels
            labels = self._generate_document_labels(example, spans)
            
            # Create thread information
            thread_info = self._create_thread_info(example)
            
            # Create final DLP format
            dlp_example = {
                "channel": getattr(example, 'channel', 'email'),
                "user": getattr(example, 'user', {}),
                "recipients": getattr(example, 'recipients', []),
                "thread": thread_info,
                "subject": getattr(example, 'subject', ''),
                "body": body,
                "attachments": getattr(example, 'attachments', []),
                "links": getattr(example, 'links', []),
                "labels": labels,
                "spans": spans,
                "meta": self._create_metadata(example)
            }
            
            return dlp_example
            
        except Exception as e:
            print(f"Failed to convert example to DLP format: {e}")
            return None
    
    def _generate_document_labels(self, example: Any, spans: List[Dict]) -> Dict:
        """Generate document-level labels based on content and context."""
        labels = {
            "sensitivity": 0,
            "exposure": 0,
            "context": 1,
            "obfuscation": 0
        }
        
        # Determine sensitivity based on spans
        sensitive_span_types = ["PAN", "SSN", "SECRET", "DBURI"]
        has_sensitive_spans = any(
            span["type"] in sensitive_span_types for span in spans
        )
        if has_sensitive_spans:
            labels["sensitivity"] = 1
        
        # Determine exposure based on recipients
        recipients = getattr(example, 'recipients', [])
        personal_domains = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com"}
        
        for recipient in recipients:
            if any(domain in recipient for domain in personal_domains):
                labels["exposure"] = 1
                labels["context"] = 0  # Inappropriate context
                break
        
        # Check for obfuscation techniques
        body = getattr(example, 'body', '')
        if self._has_obfuscation(body):
            labels["obfuscation"] = 1
        
        return labels
    
    def _has_obfuscation(self, content: str) -> bool:
        """Check if content contains obfuscation techniques."""
        techniques = self.span_detector.identify_obfuscation_techniques(content)
        return len(techniques) > 0
    
    def _create_thread_info(self, example: Any) -> Dict:
        """Create thread information for the example."""
        # Use existing thread info if available
        if hasattr(example, 'thread') and example.thread:
            return example.thread
        
        # Generate new thread info
        thread_id = getattr(example, 'thread_id', None)
        if not thread_id:
            thread_id = hash_string(f"single_{random.randint(1000, 9999)}")
        
        thread_turn = getattr(example, 'thread_turn', 1)
        
        return {
            "id_hash": thread_id,
            "age_days": random.randint(0, 30),
            "prior_msgs": max(0, thread_turn - 1)
        }
    
    def _create_metadata(self, example: Any) -> Dict:
        """Create metadata for the example."""
        body = getattr(example, 'body', '')
        
        # Basic metadata
        meta = {
            "base64": "base64" in body.lower(),
            "homoglyph": any(ord(c) > 127 for c in body),
            "ts": generate_timestamp(),
            "agentic_generated": True
        }
        
        # Add existing metadata if available
        if hasattr(example, 'meta') and example.meta:
            meta.update(example.meta)
        
        # Add generation metadata if available
        if hasattr(example, 'generation_metadata') and example.generation_metadata:
            meta["generator_agent"] = example.generation_metadata.get("agent", "unknown")
        
        # Add quality score if available
        if hasattr(example, 'quality_score'):
            meta["quality_score"] = example.quality_score
            
        return meta
    
    def batch_convert(self, examples: List[Any]) -> List[Dict]:
        """Convert multiple examples to DLP format."""
        converted = []
        for example in examples:
            dlp_example = self.convert_to_dlp_format(example)
            if dlp_example:
                converted.append(dlp_example)
        return converted