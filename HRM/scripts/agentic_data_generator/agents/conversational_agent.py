"""
Conversational Agent - Tier 3: Simulate realistic multi-turn conversations.
"""

import json
import hashlib
import time
from typing import List, Optional
from .base_agent import BaseLLMAgent, GenerationRequest, GeneratedExample


class ConversationalAgent(BaseLLMAgent):
    """Tier 3: Simulate realistic multi-turn conversations."""
    
    def __init__(self, config):
        super().__init__(config, "Conversational")
        self.active_threads = {}
    
    async def generate_conversation_thread(self, request: GenerationRequest) -> List[GeneratedExample]:
        """Generate a multi-turn conversation thread."""
        if request.conversation_turns <= 1:
            return []
        
        thread_id = hashlib.md5(f"{request.agent_type}_{request.scenario_context}_{time.time()}".encode()).hexdigest()[:8]
        
        conversation_prompt = f"""Simulate a realistic business email thread between 2-3 people.

Base Scenario: {request.scenario_context}
Risk Level: {request.risk_level}
Target Spans: {request.target_spans}
Thread Length: {request.conversation_turns} messages

Create a realistic email thread where:
1. People respond naturally to each other
2. Information builds up over the conversation
3. Sensitive information appears organically 
4. Same people maintain consistent email addresses and roles
5. Each message feels like a natural response to the previous

Generate {request.conversation_turns} JSON objects representing the thread chronologically:

{{
  "channel": "email",
  "user": {{"role": "APPROPRIATE_ROLE", "dept": "RELEVANT_DEPT", "seniority": "CONTEXTUAL_LEVEL"}},
  "recipients": ["naturally.generated@domain.com"],
  "subject": "Re: Realistic subject that builds on the conversation",
  "body": "Response building on previous message with new information...",
  "attachments": [],
  "links": [],
  "thread_turn": 1
}}

Return one JSON per line. Make the conversation feel natural and realistic."""

        system_prompt = "You are simulating realistic workplace email conversations. People should respond naturally, build on previous messages, and maintain consistent personas."
        
        response = await self._generate_with_llm(conversation_prompt, system_prompt)
        if response:
            examples = self._parse_conversation_response(response, thread_id)
            if examples:
                self.generation_stats["successful"] += len(examples)
                return examples
        
        self.generation_stats["failed"] += request.conversation_turns
        return []
    
    def _parse_conversation_response(self, response: str, thread_id: str) -> List[GeneratedExample]:
        """Parse multi-JSON response into conversation examples."""
        examples = []
        lines = response.strip().split('\\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    
                    # Create thread info
                    thread_data = {"id_hash": thread_id, "prior_msgs": i}
                    meta_data = {"agent": "conversational"}
                    
                    example = GeneratedExample(
                        channel=data.get("channel", "email"),
                        user=data.get("user", {}),
                        recipients=data.get("recipients", []),
                        subject=data.get("subject", ""),
                        body=data.get("body", ""),
                        attachments=data.get("attachments", []),
                        links=data.get("links", []),
                        thread=thread_data,
                        labels=data.get("labels"),
                        spans=data.get("spans"),
                        meta=meta_data
                    )
                    examples.append(example)
                    
                except json.JSONDecodeError:
                    continue
        
        return examples
    
    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        """Generate single example (used by coordinator)."""
        thread = await self.generate_conversation_thread(request)
        return thread[0] if thread else None