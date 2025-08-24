"""
HRM-DLP: Hierarchical Reasoning Model for Data Loss Prevention

A clean, simplified implementation of the HRM architecture with DLP extensions
for email/chat content analysis, PII detection, and trust scoring.
"""

from . import hrm
from . import dlp
from . import data
from . import utils

__version__ = "1.0.0"
__all__ = ["hrm", "dlp", "data", "utils"]