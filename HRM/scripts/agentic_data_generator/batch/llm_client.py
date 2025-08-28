"""
LLM client management with simplified initialization and model selection.
"""

import os
import random
from typing import Dict, Tuple, Optional

# Import LLM clients with availability checks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False


class LLMClientManager:
    """Manages LLM clients and model selection."""
    
    def __init__(self, config=None):
        self.config = config
        self.clients = self._initialize_clients()
        self.model_weights = self._get_model_weights()
    
    def _initialize_clients(self) -> Dict:
        """Initialize available LLM clients."""
        clients = {}
        
        # OpenAI and compatible providers
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            clients["openai"] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # HuggingFace Inference (OpenAI-compatible)
            if os.getenv("HUGGINGFACE_API_KEY"):
                clients["huggingface_inference"] = openai.OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=os.getenv("HUGGINGFACE_API_KEY")
                )
                
            # Fireworks AI (OpenAI-compatible) 
            if os.getenv("FIREWORKS_API_KEY"):
                clients["fireworks"] = openai.OpenAI(
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.getenv("FIREWORKS_API_KEY")
                )
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            clients["anthropic"] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Together AI
        if TOGETHER_AVAILABLE and os.getenv("TOGETHER_API_KEY"):
            clients["together"] = Together(api_key=os.getenv("TOGETHER_API_KEY"))
            
        return clients
    
    def _get_model_weights(self) -> Dict:
        """Get model weights from config or use defaults."""
        if self.config and hasattr(self.config, 'model_names'):
            return self.config.model_names
            
        # Default model configuration - ONLY models that support structured outputs
        # Prioritize GPT-5 for HRM-DLP generation quality
        return {
            "openai": {
                "gpt-5-2025-08-07": 5,           # Highest priority for HRM-DLP
                "gpt-5-mini-2025-08-07": 4,      # Good balance of quality/cost
                "gpt-4o-2024-08-06": 2,          # Fallback option
                "gpt-4o-mini-2024-07-18": 1,     # Last resort
            },
            "anthropic": {
                # Prefer widely-available models to avoid 404 on older accounts
                "claude-3-haiku-20240307": 3
            },
        }
    
    def get_available_providers(self) -> list:
        """Get list of available providers."""
        return list(self.clients.keys())
    
    def choose_model(self, provider_preference: Optional[str] = None) -> Tuple[str, str]:
        """Choose a provider and model, optionally with provider preference."""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            raise ValueError("No LLM providers available")
        
        # Use preferred provider if available
        if provider_preference and provider_preference in available_providers:
            provider = provider_preference
        else:
            provider = random.choice(available_providers)
        
        # Choose model for the provider
        if provider in self.model_weights:
            models_with_weights = self.model_weights[provider]
            models = []
            for model, weight in models_with_weights.items():
                models.extend([model] * weight)
            model = random.choice(models)
        else:
            # Fallback to default models
            default_models = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-haiku-20240307",
            }
            model = default_models.get(provider, "gpt-4o")
        
        return provider, model
    
    def get_client(self, provider: str):
        """Get client for specific provider."""
        return self.clients.get(provider)
    
    def get_model_temperature(self, model: str) -> float:
        """Get appropriate temperature for model."""
        # Models that require temperature=1.0
        restricted_models = [
            "gpt-5-", "o1-", "o3-", "o4-", "gpt-4.1-",
            "chatgpt-4o-latest", "gpt-4-turbo"
        ]
        
        if any(restricted in model for restricted in restricted_models):
            return 1.0
            
        # Use config temperature or default
        if self.config and hasattr(self.config, 'temperature'):
            return self.config.temperature
        
        return 0.8  # Default temperature
    
    def supports_structured_outputs(self, provider: str, model: str) -> bool:
        """Check if provider/model combination supports structured outputs (json_schema)."""
        if provider == "openai":
            # OpenAI models that support structured outputs
            structured_output_models = {
                "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07",
                "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"
            }
            return model in structured_output_models
            
        # Anthropic doesn't use structured outputs, but handles JSON differently
        return provider == "anthropic"

    def is_batch_compatible(self, provider: str, model: str) -> bool:
        """Check if provider/model combination supports batch processing."""
        if provider == "openai":
            # Only allow models that support structured outputs for batching
            return self.supports_structured_outputs(provider, model) and model != "chatgpt-4o-latest"
            
        if provider == "anthropic":
            # Anthropic batchable models
            anthropic_batchable = {
                "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", 
                "claude-3-5-haiku-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"
            }
            return model in anthropic_batchable
        
        # Other providers don't support batching
        return False
