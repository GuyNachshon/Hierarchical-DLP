"""
Agent classes for the 3-tier agentic data generation system.
"""

from .base_agent import BaseLLMAgent
from .manager_agent import ManagerAgent
from .domain_agents import LegalAgent, FinanceAgent, HRAgent, SecurityAgent
from .specialized_agents import CasualAgent, CleanBusinessAgent, ObfuscationSpecialist
from .conversational_agent import ConversationalAgent
from .augmentation_agent import AugmentationAgent

__all__ = [
    "BaseLLMAgent",
    "ManagerAgent", 
    "LegalAgent",
    "FinanceAgent",
    "HRAgent", 
    "SecurityAgent",
    "CasualAgent",
    "CleanBusinessAgent",
    "ObfuscationSpecialist",
    "ConversationalAgent",
    "AugmentationAgent"
]
