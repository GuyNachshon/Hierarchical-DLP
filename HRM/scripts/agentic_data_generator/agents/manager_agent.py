"""
Manager Agent - Tier 1: Controls dataset orchestration and quality evaluation.
"""

import json
import random
from typing import List, Optional, Dict
from .base_agent import BaseLLMAgent, GenerationRequest, GeneratedExample
from utils.patterns import SpanDetector


class GenerationRequestSchema:
    """Simple schema for generation requests."""
    
    def __init__(self, agent_type: str, risk_level: str, scenario_context: str,
                 target_spans: List[str], conversation_turns: int = 1,
                 thread_context: Optional[Dict] = None, count: int = 1):
        self.agent_type = agent_type
        self.risk_level = risk_level
        self.scenario_context = scenario_context
        self.target_spans = target_spans
        self.conversation_turns = conversation_turns
        self.thread_context = thread_context
        self.count = count


class GenerationPlanSchema:
    """Simple schema for generation plans."""
    
    def __init__(self, total_examples: int, requests: List[GenerationRequestSchema]):
        self.total_examples = total_examples
        self.requests = requests


class ManagerAgent(BaseLLMAgent):
    """Tier 1: Manager Agent - Controls dataset orchestration."""
    
    def __init__(self, config):
        super().__init__(config, "Manager")
        self.span_detector = SpanDetector()
        self.dataset_stats = {
            "agent_counts": {agent: 0 for agent in config.agent_distribution.keys()},
            "risk_counts": {risk: 0 for risk in config.risk_distribution.keys()},
            "span_type_counts": {},
            "quality_scores": []
        }
    
    async def create_generation_plan(self, split: str, target_size: int) -> List[GenerationRequest]:
        """Create detailed generation plan for a dataset split."""
        planning_prompt = f"""Create a sophisticated generation plan for {target_size} examples in the {split} split for DLP training data.

Your goals:
1. Balance risk levels: {self.config.risk_distribution}
2. Balance agent types: {self.config.agent_distribution}
3. Ensure diverse span types (EMAIL, PAN, SSN, SECRET, NDA, DBURI, etc.)
4. Create realistic business scenarios with detailed contexts
5. Include {int(target_size * self.config.thread_probability)} multi-turn conversations
6. Generate scenarios that reflect real workplace communications

For each request, create detailed scenario contexts that include:
- Specific business situations (e.g., "NDA review with external counsel for merger evaluation")
- Realistic risk scenarios (e.g., "payment authorization shared without dual approval") 
- Natural business contexts that justify the communication
- Appropriate target spans for each scenario type

Focus on realism and diversity - these should feel like actual workplace communications that DLP systems would encounter."""
        
        system_prompt = "You are an expert dataset manager creating balanced, diverse, realistic training data for DLP systems. Focus on generating sophisticated business scenarios that reflect real workplace communications with varying levels of data sensitivity and risk."
        
        # Try structured generation first
        client = self.clients.get("openai")
        if client:
            try:
                response = await self._generate_with_llm(planning_prompt, system_prompt)
                if response:
                    return self._parse_generation_plan(response, target_size)
            except Exception as e:
                print(f"[Manager] Structured generation failed: {e}")
        
        # Fallback to rule-based planning
        return self._create_fallback_plan(target_size)
    
    def _parse_generation_plan(self, response: str, target_size: int) -> List[GenerationRequest]:
        """Parse LLM response into generation requests."""
        try:
            plan_data = self._extract_json_from_response(response)
            if plan_data and "requests" in plan_data:
                requests = []
                for req_data in plan_data["requests"]:
                    requests.append(GenerationRequest(
                        agent_type=req_data.get("agent_type", "casual"),
                        risk_level=req_data.get("risk_level", "medium_risk"),
                        scenario_context=req_data.get("scenario_context", "general"),
                        target_spans=req_data.get("target_spans", []),
                        conversation_turns=req_data.get("conversation_turns", 1),
                        thread_context=req_data.get("thread_context"),
                        count=req_data.get("count", 1)
                    ))
                return requests
        except Exception as e:
            print(f"[Manager] Failed to parse generation plan: {e}")
            
        return self._create_fallback_plan(target_size)
    
    def _create_fallback_plan(self, target_size: int) -> List[GenerationRequest]:
        """Fallback generation plan using rule-based approach with sophisticated scenarios."""
        requests = []
        agents = list(self.config.agent_distribution.keys())
        risks = list(self.config.risk_distribution.keys())
        
        for i in range(target_size):
            agent = random.choices(agents, weights=list(self.config.agent_distribution.values()))[0]
            risk = random.choices(risks, weights=list(self.config.risk_distribution.values()))[0]
            
            # Generate sophisticated scenario context based on agent and risk
            scenario_context = self._generate_sophisticated_scenario(agent, risk)
            
            # Determine target spans based on agent type
            span_options = {
                "legal": ["NDA", "NAME", "EMAIL"],
                "finance": ["PAN", "SSN", "PHONE"],
                "hr": ["SSN", "NAME", "PHONE"],
                "security": ["SECRET", "DBURI"],
                "casual": ["EMAIL", "NAME"],
                "clean_business": [],  # No sensitive spans
                "obfuscation": ["PAN", "EMAIL"]  # For obfuscation testing
            }
            
            target_spans = span_options.get(agent, ["EMAIL", "NAME"])
            if agent != "clean_business":
                target_spans = random.sample(target_spans, k=min(2, len(target_spans)))
            
            conversation_turns = 1
            if random.random() < self.config.thread_probability:
                conversation_turns = random.randint(2, 4)
            
            requests.append(GenerationRequest(
                agent_type=agent,
                risk_level=risk,
                scenario_context=scenario_context,
                target_spans=target_spans,
                conversation_turns=conversation_turns
            ))
        
        return requests
    
    def evaluate_example_quality(self, example: GeneratedExample) -> float:
        """Evaluate generated example quality (0-1 score)."""
        score = 0.0
        
        # Content quality checks
        body = example.body
        if len(body) >= 50 and len(body) <= 2000:
            score += 0.2
        
        if not any(word in body.lower() for word in ["lorem", "placeholder", "example"]):
            score += 0.2
        
        # Span validation
        spans = self.span_detector.extract_all_spans(body)
        if 1 <= len(spans) <= 5:
            score += 0.3
        
        # Realism checks
        if example.subject and len(example.subject) > 5:
            score += 0.1
        
        if example.recipients and all("@" in r for r in example.recipients):
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_sophisticated_scenario(self, agent_type: str, risk_level: str) -> str:
        """Generate sophisticated, realistic business scenarios based on agent type and risk level."""
        
        scenarios = {
            "legal": {
                "no_risk": [
                    "NDA review with external counsel at established law firm for routine merger evaluation",
                    "Contract negotiation coordination with client's authorized legal representatives",
                    "Legal matter discussion with co-counsel regarding ongoing litigation strategy",
                    "Compliance review coordination with regulatory counsel for standard audit",
                    "IP licensing negotiation with approved external legal advisors"
                ],
                "medium_risk": [
                    "Confidential legal strategy sharing with external counsel without proper verification",
                    "Client privileged information discussion via unsecured communication channels",
                    "Legal document review containing sensitive merger details sent to wrong recipient",
                    "Attorney work product shared with opposing counsel by administrative error",
                    "Settlement negotiation details accidentally copied to unauthorized parties"
                ],
                "high_risk": [
                    "Privileged legal communications forwarded to personal email accounts",
                    "Confidential legal strategy leaked to competitors through insecure channels",
                    "Client confidential information shared with unauthorized external parties",
                    "Legal settlement terms disclosed to media contacts without authorization",
                    "Attorney-client privileged documents sent to opposing parties"
                ]
            },
            "finance": {
                "no_risk": [
                    "Payment processing coordination with established banking partners for routine transactions",
                    "Invoice submission to approved vendor payment portals within authorization limits",
                    "Treasury operations communication with designated banking relationship managers",
                    "Accounts payable processing through verified supplier payment systems",
                    "Financial reporting coordination with authorized external auditing firms"
                ],
                "medium_risk": [
                    "Payment authorization details shared without proper dual approval verification",
                    "Banking credentials communicated through unsecured internal messaging systems",
                    "Financial data transmission to vendors without proper encryption protocols",
                    "Wire transfer instructions sent via email without secondary confirmation",
                    "Account reconciliation data shared with unauthorized finance personnel"
                ],
                "high_risk": [
                    "Corporate banking credentials leaked through insecure communication channels",
                    "Payment card data transmitted to unauthorized external financial institutions",
                    "Treasury account information disclosed to non-approved third-party vendors",
                    "Financial reporting data containing sensitive metrics sent to competitors",
                    "Banking API keys shared with unauthorized fintech integration partners"
                ]
            },
            "hr": {
                "no_risk": [
                    "Employee onboarding coordination with authorized HR systems and verified personnel",
                    "Benefits enrollment processing through approved third-party administrators",
                    "Performance review scheduling with designated HR business partners",
                    "Training program coordination with certified external learning providers",
                    "Payroll processing communication with authorized payroll service vendors"
                ],
                "medium_risk": [
                    "Employee personal information shared without proper consent verification",
                    "Performance review data transmitted through unsecured HR communication channels",
                    "Benefits enrollment details sent to wrong HR service provider contact",
                    "Employee compensation information discussed with unauthorized managers",
                    "Personnel records access granted without following proper approval protocols"
                ],
                "high_risk": [
                    "Employee SSN and personal data leaked through insecure external HR platforms",
                    "Confidential personnel records disclosed to unauthorized third-party vendors",
                    "Employee disciplinary information shared with non-approved external parties",
                    "Payroll data containing sensitive compensation details sent to wrong recipients",
                    "Employee personal information forwarded to unauthorized recruitment agencies"
                ]
            },
            "security": {
                "no_risk": [
                    "API key rotation coordination with authorized development teams through secure channels",
                    "Database access provisioning for approved personnel following security protocols",
                    "Security incident response coordination with designated incident response partners",
                    "Vulnerability assessment planning with certified security consulting firms",
                    "Access control review with authorized IT security personnel"
                ],
                "medium_risk": [
                    "Database connection strings shared without proper access control verification",
                    "API credentials communicated through unsecured internal development channels",
                    "Security configuration details transmitted without following change management",
                    "System access tokens shared with development teams lacking proper clearance",
                    "Infrastructure credentials discussed in unsecured team communication platforms"
                ],
                "high_risk": [
                    "Production database credentials leaked through insecure development communication",
                    "Critical system API keys shared with unauthorized external development partners",
                    "Security incident details containing sensitive system information disclosed publicly",
                    "Infrastructure access tokens transmitted to non-approved third-party vendors",
                    "Database connection strings exposed through insecure code repository commits"
                ]
            },
            "casual": {
                "no_risk": [
                    "Team project updates and milestone coordination through approved collaboration platforms",
                    "Meeting scheduling and agenda distribution to verified internal team members",
                    "General business communication with established external partners and clients",
                    "Customer support coordination through authorized customer service channels",
                    "Sales outreach communication with prospects using approved marketing platforms"
                ],
                "medium_risk": [
                    "Project information accidentally shared with external parties without authorization review",
                    "Customer contact details transmitted through unsecured team communication channels",
                    "Internal meeting notes containing sensitive discussions forwarded inappropriately",
                    "Business strategy details discussed in unsecured external communication platforms",
                    "Client information shared with team members lacking proper business need"
                ],
                "high_risk": [
                    "Confidential business strategy leaked to competitors through personal communication",
                    "Customer personal information disclosed to unauthorized external marketing vendors",
                    "Internal project details containing proprietary information sent to wrong recipients",
                    "Business development discussions with sensitive partnership details exposed publicly",
                    "Client confidential information forwarded to personal email accounts"
                ]
            },
            "clean_business": {
                "no_risk": [
                    "General project status updates and coordination without any sensitive information",
                    "Public meeting scheduling and non-confidential agenda distribution",
                    "General business communication about non-sensitive operational topics",
                    "Public marketing material review and feedback coordination",
                    "Non-confidential team building and administrative communication"
                ]
            },
            "obfuscation": {
                "no_risk": [
                    "Business communication using semantic obfuscation for authorized information sharing",
                    "Approved external communication using euphemisms for legitimate business purposes",
                    "Authorized information sharing using indirect reference methods"
                ],
                "medium_risk": [
                    "Sensitive information sharing using basic obfuscation techniques without authorization",
                    "Business data transmitted using semantic hiding methods to bypass security controls",
                    "Financial information communicated using coded language to unauthorized recipients"
                ],
                "high_risk": [
                    "Critical business secrets shared using sophisticated obfuscation to evade detection",
                    "Sensitive personal information leaked using semantic techniques to bypass DLP systems",
                    "Confidential corporate data transmitted using advanced hiding methods to unauthorized parties"
                ]
            }
        }
        
        # Get scenarios for the agent type and risk level
        agent_scenarios = scenarios.get(agent_type, scenarios["casual"])
        risk_scenarios = agent_scenarios.get(risk_level, agent_scenarios.get("no_risk", ["General business communication"]))
        
        if not risk_scenarios:
            risk_scenarios = [f"General {agent_type} communication with {risk_level} risk level"]
        
        return random.choice(risk_scenarios)
    
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        """Manager doesn't generate examples directly."""
        raise NotImplementedError("Manager coordinates but doesn't generate examples")