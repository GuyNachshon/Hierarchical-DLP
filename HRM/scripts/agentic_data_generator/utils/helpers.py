"""
Helper utilities for DLP data generation.
"""

import hashlib
import random
import time
from datetime import datetime, timedelta
from typing import Any


def generate_timestamp() -> str:
    """Generate a realistic timestamp for training data."""
    # Generate timestamp within last 90 days
    now = datetime.now()
    days_ago = random.randint(0, 90)
    timestamp = now - timedelta(days=days_ago)
    
    # Add some random time variation
    hours = random.randint(8, 18)  # Business hours
    minutes = random.randint(0, 59)
    timestamp = timestamp.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    
    return timestamp.isoformat()


def hash_string(text: str) -> str:
    """Generate a hash for thread IDs and other identifiers."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def calculate_quality_score(example: Any) -> float:
    """Calculate quality score for a generated example."""
    score = 0.5  # Base score
    
    # Check body content
    if hasattr(example, 'body') and example.body:
        body = example.body
        
        # Content length check
        if 50 <= len(body) <= 2000:
            score += 0.2
            
        # Avoid placeholder text
        if not any(word in body.lower() for word in ["lorem", "placeholder", "example"]):
            score += 0.1
            
        # Check for realistic content
        business_words = ["project", "meeting", "client", "team", "update", "review"]
        if any(word in body.lower() for word in business_words):
            score += 0.1
    
    # Check subject
    if hasattr(example, 'subject') and example.subject and len(example.subject) > 5:
        score += 0.1
        
    # Check recipients
    if hasattr(example, 'recipients') and example.recipients:
        if all("@" in recipient for recipient in example.recipients):
            score += 0.1
    
    return min(score, 1.0)


def validate_email_format(email: str) -> bool:
    """Simple email format validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    cleaned = ' '.join(text.split())
    
    # Remove control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\t')
    
    return cleaned.strip()


def truncate_text(text: str, max_length: int = 2000) -> str:
    """Truncate text to maximum length with proper word boundaries."""
    if len(text) <= max_length:
        return text
        
    truncated = text[:max_length]
    
    # Find the last complete word
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Only truncate if we don't lose too much
        truncated = truncated[:last_space]
    
    return truncated + "..."