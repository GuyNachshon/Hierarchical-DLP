"""
Base agent class for LLM-based data generation.
"""

import os
import random
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Any

# Import LLM clients (with availability checks)
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


class GeneratedExample:
    """Simple data class for generated examples."""
    
    def __init__(self, channel: str, user: Dict, recipients: List[str], 
                 subject: str, body: str, attachments: List[str] = None,
                 links: List[str] = None, thread: Optional[Dict] = None,
                 labels: Optional[Dict] = None, spans: Optional[List] = None,
                 meta: Optional[Dict] = None):
        self.channel = channel
        self.user = user
        self.recipients = recipients or []
        self.subject = subject
        self.body = body
        self.attachments = attachments or []
        self.links = links or []
        self.thread = thread
        self.labels = labels
        self.spans = spans
        self.meta = meta or {}
        
        # Legacy compatibility
        self.thread_id = thread.get('id_hash') if thread else None
        self.thread_turn = thread.get('prior_msgs', 0) + 1 if thread else 1
        self.quality_score = 0.0
        self.generation_metadata = meta or {}


class GenerationRequest:
    """Request object for content generation."""
    
    def __init__(self, agent_type: str, risk_level: str, scenario_context: str,
                 target_spans: List[str], thread_context: Optional[Dict] = None,
                 conversation_turns: int = 1, count: int = 1):
        self.agent_type = agent_type
        self.risk_level = risk_level
        self.scenario_context = scenario_context
        self.target_spans = target_spans
        self.thread_context = thread_context
        self.conversation_turns = conversation_turns
        self.count = count


class BaseLLMAgent(ABC):
    """Base class for all LLM agents with simplified client management."""
    
    def __init__(self, config: Any, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.clients = self._init_llm_clients()
        self.generation_stats = {"successful": 0, "failed": 0}
    
    def _init_llm_clients(self) -> Dict:
        """Initialize LLM clients based on available libraries and API keys."""
        clients = {}
        
        # OpenAI
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
    
    def _choose_model(self) -> Tuple[str, str]:
        """Choose a random model from available providers."""
        # Default model selections with weights
        model_choices = {}
        
        if "openai" in self.clients:
            model_choices["openai"] = [
                ("gpt-4o", 3),
                ("gpt-4o-mini", 2),
                ("gpt-3.5-turbo", 1)
            ]
            
        if "anthropic" in self.clients:
            model_choices["anthropic"] = [
                ("claude-3-haiku-20240307", 2),
                ("claude-3-sonnet-20240229", 1)
            ]
            
        if "together" in self.clients:
            model_choices["together"] = [
                ("meta-llama/Llama-2-7b-chat-hf", 1)
            ]
        
        # Choose provider
        if not model_choices:
            raise ValueError("No LLM clients available")
        
        # Prefer reliable providers for planning/generation
        available_providers = list(model_choices.keys())
        if "openai" in available_providers:
            provider = "openai"
        elif "anthropic" in available_providers:
            provider = "anthropic"
        else:
            provider = random.choice(available_providers)
        
        # Choose model with weights
        models_with_weights = model_choices[provider]
        models = [model for model, weight in models_with_weights for _ in range(weight)]
        model = random.choice(models)
        
        return provider, model
    
    def _get_model_temperature(self, model: str) -> float:
        """Get appropriate temperature for model."""
        # Some models only support temperature=1.0
        restricted_models = [
            "gpt-5-", "o1-", "o3-", "o4-", "gpt-4.1-",
            "chatgpt-4o-latest", "gpt-4-turbo"
        ]
        
        if any(restricted in model for restricted in restricted_models):
            return 1.0
            
        return getattr(self.config, 'temperature', 0.8)
    
    def _get_structured_output_schema(self) -> Dict:
        """Get JSON schema for structured output validation."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hrm_dlp_example",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "enum": ["email", "chat", "pr", "upload"]
                        },
                        "user": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "dept": {"type": "string"},
                                "seniority": {"type": "string"}
                            },
                            "required": ["role", "dept", "seniority"],
                            "additionalProperties": False
                        },
                        "recipients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "attachments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "size": {"type": "integer", "minimum": 1},
                                    "mime_type": {"type": "string"},
                                    "content_summary": {"type": "string"},
                                    "sensitivity_indicators": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "minItems": 1
                                    }
                                },
                                "required": ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"],
                                "additionalProperties": False
                            },
                            "minItems": 1
                        },
                        "links": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "context_summary": {"type": "string"},
                        "thread": {
                            "type": "object",
                            "properties": {
                                "id_hash": {"type": "string"},
                                "age_days": {"type": "integer"},
                                "prior_msgs": {"type": "integer"}
                            },
                            "required": ["id_hash", "age_days", "prior_msgs"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["channel", "user", "recipients", "subject", "body", "attachments", "links", "context_summary", "thread"],
                    "additionalProperties": False
                }
            }
        }
    
    async def _generate_with_llm(self, prompt: str, system: str) -> Optional[str]:
        """Generate content using available LLM with simplified retry logic."""
        provider, model = self._choose_model()
        client = self.clients.get(provider)
        
        if not client:
            return None
            
        max_retries = getattr(self.config, 'max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                temperature = self._get_model_temperature(model)
                
                if provider == "anthropic":
                    response = await asyncio.to_thread(
                        client.messages.create,
                        model=model,
                        max_tokens=4096,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                    
                elif provider in ["openai", "huggingface_inference", "fireworks", "together"]:
                    # Prepare request parameters
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature
                    }
                    
                    # Add structured output for OpenAI models that support it
                    # Include GPT-4o, GPT-4-turbo, and newer GPT-5 models  
                    if provider == "openai" and ("gpt-4o" in model or "gpt-4-turbo" in model or "gpt-5" in model):
                        request_params["response_format"] = self._get_structured_output_schema()
                        print(f"[DEBUG] Applied structured output schema to {provider}/{model}")
                    
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        **request_params
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                print(f"[{self.agent_name}] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response using simple pattern matching."""
        import json
        import re
        
        # Look for JSON in response
        json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try JSON code blocks
        json_blocks = re.findall(r'```(?:json)?\\n(.*?)\\n```', response, re.DOTALL)
        for block in json_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
                
        return None
    
    def build_prompts(self, request: GenerationRequest) -> Tuple[str, str]:
        """Build (system, prompt) pair for generation request with sophisticated prompts."""
        # Generate sophisticated system prompt based on agent type
        system_prompts = {
            "legal": "You are an expert legal professional with 15+ years of corporate law experience. Generate realistic legal communications that lawyers would actually write, including appropriate legal terminology and business context.",
            "finance": "You are an expert finance professional with deep knowledge of corporate payments, banking, and financial operations. Generate realistic financial communications with proper business context.",
            "hr": "You are an expert HR professional with deep knowledge of employee data management, privacy laws, and HR processes. Generate realistic HR communications.",
            "security": "You are an expert cybersecurity professional with deep knowledge of credential management, security incidents, and DevOps practices. Generate realistic security communications.",
            "casual": "You are simulating everyday workplace communication. Generate realistic, natural business emails that various employees might write.",
            "clean_business": "You are generating completely clean workplace communication. The content must be realistic business communication that contains absolutely no sensitive information, personal data, secrets, or confidential content.",
            "obfuscation": "You are an expert at sophisticated information hiding techniques. Create business communications that contain semantically obfuscated sensitive information using natural business context."
        }
        
        system_prompt = system_prompts.get(request.agent_type, f"You are a {request.agent_type} domain expert with deep professional experience.")
        
        # Generate sophisticated user prompt based on agent type and scenario
        agent_guidelines = {
            "legal": self._get_legal_guidelines(request),
            "finance": self._get_finance_guidelines(request),
            "hr": self._get_hr_guidelines(request), 
            "security": self._get_security_guidelines(request),
            "casual": self._get_casual_guidelines(request),
            "clean_business": self._get_clean_business_guidelines(request),
            "obfuscation": self._get_obfuscation_guidelines(request)
        }
        
        base_prompt = f"""Generate realistic {request.agent_type} communication that naturally includes the required sensitive information.

Scenario: {request.scenario_context}
Risk Level: {request.risk_level}
Required Spans: {request.target_spans}
Conversation Turns: {request.conversation_turns}

{agent_guidelines.get(request.agent_type, "Generate realistic business communication.")}

Generate JSON with this exact structure:
{{
  "channel": "email",
  "user": {{"role": "APPROPRIATE_ROLE", "dept": "RELEVANT_DEPT", "seniority": "CONTEXTUAL_LEVEL"}},
  "recipients": ["realistic@domain.com"],
  "subject": "Realistic subject based on scenario", 
  "body": "Naturally written communication with embedded sensitive data...",
  "attachments": [
    {{
      "name": "relevant_filename.docx",
      "size": 245760,
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "content_summary": "Detailed description of what sensitive information the file contains",
      "sensitivity_indicators": ["relevant_sensitivity_type", "additional_indicator"]
    }}
  ],
  "links": ["https://realistic.domain.com/path"]
}}"""
        
        return system_prompt, base_prompt
    
    def _get_legal_guidelines(self, request: GenerationRequest) -> str:
        """Get legal-specific guidelines for prompt generation."""
        return """Legal context guidelines:
- AUTHORIZED: NDA review with external counsel at established law firms
- AUTHORIZED: Legal matter coordination with co-counsel and authorized representatives
- AUTHORIZED: Contract negotiation with client's legal team
- For leak scenarios: Mix legal content with personal/unauthorized recipients
- Use realistic legal terminology and business justification
- Include attachment references: legal_brief.pdf, nda_draft.docx, contract_terms.pdf
- Match role/dept/seniority to scenario context naturally"""
    
    def _get_finance_guidelines(self, request: GenerationRequest) -> str:
        """Get finance-specific guidelines for prompt generation."""
        return """Finance context guidelines:
- AUTHORIZED: Payment processing with established banking partners
- AUTHORIZED: Invoice submission to vendor payment portals  
- AUTHORIZED: Treasury operations with banking relationships
- For leak scenarios: Mix financial data with unauthorized recipients
- Include realistic financial data and business context
- Reference attachments: payment_details.xlsx, wire_instructions.docx, invoice_backup.pdf
- Use appropriate finance roles: FINANCE, ACCOUNTING, TREASURY, CONTROLLER"""
    
    def _get_hr_guidelines(self, request: GenerationRequest) -> str:
        """Get HR-specific guidelines for prompt generation."""
        return """HR context guidelines:
- Employee onboarding and personal data collection scenarios
- Benefits enrollment and personal information updates
- Performance reviews and disciplinary actions
- For leak scenarios: Mix HR data with unauthorized recipients  
- Include realistic employee information and business context
- Reference attachments: employee_data.xlsx, benefits_enrollment.pdf, performance_review.docx
- Use appropriate HR roles: HR, PEOPLE, RECRUITING, BENEFITS"""
    
    def _get_security_guidelines(self, request: GenerationRequest) -> str:
        """Get security-specific guidelines for prompt generation."""
        return """Security context guidelines:
- API key sharing and credential management scenarios
- Database access and connection strings
- Security incident responses and investigations
- For leak scenarios: Mix security data with unauthorized recipients
- Include realistic technical context and security procedures  
- Reference attachments: access_logs.txt, incident_report.pdf, security_config.json
- Use appropriate security roles: SECURITY, IT, DEVOPS, INFRASTRUCTURE"""
    
    def _get_casual_guidelines(self, request: GenerationRequest) -> str:
        """Get casual business guidelines for prompt generation."""
        return """Casual business context guidelines:
- Team updates and project communications
- Meeting notes and scheduling coordination
- Customer support and external communications  
- Sales and marketing outreach scenarios
- For leak scenarios: Accidentally include sensitive information in routine communications
- Use natural, everyday business language
- Reference attachments: meeting_notes.docx, project_update.pdf, customer_list.xlsx"""
    
    def _get_clean_business_guidelines(self, request: GenerationRequest) -> str:
        """Get clean business guidelines for prompt generation."""
        return """Clean business context guidelines:
CRITICAL: Generate realistic workplace communication with ZERO sensitive information:
- NO credit card numbers, SSNs, passwords, API keys, or secrets
- NO personal information beyond public business context
- NO confidential data or proprietary information
- Focus on normal, everyday workplace interactions
- Use public, non-sensitive business information only"""
    
    def _get_obfuscation_guidelines(self, request: GenerationRequest) -> str:
        """Get obfuscation-specific guidelines for prompt generation."""
        return """Obfuscation context guidelines:
Create communications where sensitive information is semantically hidden:
- NUMERICAL WORDS: "four five three two" instead of "4532"  
- TEXTUAL DECOMPOSITION: "john dot smith at company dot com"
- EUPHEMISMS: "sixteen-digit payment identifier" instead of "credit card"
- INDIRECT REFERENCES: "the account we discussed" instead of direct mentions
- Use appropriate business context to justify the obfuscation style
- Make it feel natural and business-appropriate"""
    
    def parse_result(self, raw_text: str, request: GenerationRequest) -> Optional[GeneratedExample]:
        """Convert raw LLM output into GeneratedExample object."""
        if not raw_text:
            return None
            
        parsed = self._extract_json_from_response(raw_text)
        if not parsed:
            return None
            
        try:
            return GeneratedExample(
                channel=parsed.get("channel", "email"),
                user=parsed.get("user", {"role": request.agent_type}),
                recipients=parsed.get("recipients", []),
                subject=parsed.get("subject", ""),
                body=parsed.get("body", raw_text),
                attachments=parsed.get("attachments", []),
                links=parsed.get("links", []),
                thread=parsed.get("thread"),
                labels=parsed.get("labels"),
                spans=parsed.get("spans"),
                meta={"agent_type": request.agent_type, "risk_level": request.risk_level}
            )
        except Exception:
            return None
    
    @abstractmethod
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        """Generate an example based on request - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_example")
