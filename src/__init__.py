"""
HRM-DLP: Hierarchical Reasoning Model for Data Loss Prevention

A clean, simplified implementation of the HRM architecture with DLP extensions
for email/chat content analysis, PII detection, and trust scoring.
"""

# Only import dlp for now to avoid flash attention dependencies
from . import dlp

try:
    from . import data
    from . import utils
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = ["dlp"]