"""
HRM Core Models

Base HRM architecture and configuration classes.
"""

from pathlib import Path
import sys

# Legacy import compatibility
hrm_path = Path(__file__).parent.parent.parent / "HRM"
sys.path.insert(0, str(hrm_path))

from models.hrm.hrm_act_v1 import HRMConfig

# For now, use a simplified HRM base that wraps the legacy implementation
class HRMBase:
    """Base HRM model wrapper for legacy compatibility"""
    
    def __init__(self, config: HRMConfig):
        self.config = config
        print("HRM Base model initialized (legacy compatibility wrapper)")
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("HRM training requires legacy scripts for now")

__all__ = ["HRMBase", "HRMConfig"]