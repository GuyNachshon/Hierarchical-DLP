#!/usr/bin/env python3
"""
Debug script for OpenAI structured output schema enforcement.
Tests minimal to complex schemas to isolate attachment constraint issues.
"""

import os
import asyncio
import json
import openai
from typing import Dict, Any

class StructuredOutputDebugger:
    """Debug OpenAI structured output schema enforcement."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def get_minimal_schema(self) -> Dict:
        """Most basic schema - just require one attachment object."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "minimal_test",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "attachments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"}
                                },
                                "required": ["name"],
                                "additionalProperties": False
                            },
                            "minItems": 1,
                            "maxItems": 1
                        }
                    },
                    "required": ["subject", "attachments"],
                    "additionalProperties": False
                }
            }
        }
    
    def get_rich_attachment_schema(self) -> Dict:
        """Full rich attachment schema like in our main code."""
        return {
            "type": "json_schema", 
            "json_schema": {
                "name": "rich_attachment_test",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
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
                        }
                    },
                    "required": ["subject", "attachments"],
                    "additionalProperties": False
                }
            }
        }
    
    def get_string_array_schema(self) -> Dict:
        """Schema that explicitly allows string arrays (for comparison)."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "string_array_test", 
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "attachments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        }
                    },
                    "required": ["subject", "attachments"],
                    "additionalProperties": False
                }
            }
        }
    
    async def test_schema(self, schema: Dict, model: str, prompt: str, test_name: str):
        """Test a specific schema configuration."""
        print(f"\n=== {test_name} ===")
        print(f"Model: {model}")
        print(f"Schema: {schema['json_schema']['name']}")
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates email examples with attachments."},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            print(f"✅ Success!")
            print(f"Raw response: {result}")
            
            # Parse and analyze the result
            try:
                parsed = json.loads(result)
                attachments = parsed.get("attachments", [])
                print(f"Attachments count: {len(attachments)}")
                
                if attachments:
                    first_attachment = attachments[0]
                    print(f"First attachment type: {type(first_attachment)}")
                    if isinstance(first_attachment, dict):
                        print(f"First attachment keys: {list(first_attachment.keys())}")
                        print(f"✅ SUCCESS: Generated object structure!")
                    else:
                        print(f"First attachment value: {first_attachment}")
                        print(f"❌ ISSUE: Generated string instead of object")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
        except Exception as e:
            print(f"❌ API call failed: {e}")
        
        print("-" * 50)

async def main():
    """Run structured output debugging tests."""
    debugger = StructuredOutputDebugger()
    
    # Test prompts
    simple_prompt = "Generate an email about a project update that includes one attachment."
    rich_prompt = "Generate a legal email about an NDA review that includes confidential document attachments with detailed metadata."
    
    # Test models
    models = ["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    
    print("🔍 OpenAI Structured Output Schema Debugging")
    print("=" * 60)
    
    # Phase 1: Minimal schema test
    print("\n🧪 PHASE 1: MINIMAL SCHEMA TESTING")
    minimal_schema = debugger.get_minimal_schema()
    
    for model in models:
        await debugger.test_schema(
            minimal_schema, 
            model, 
            simple_prompt,
            f"Minimal Object Schema - {model}"
        )
    
    # Phase 2: String array comparison (should work)
    print("\n🧪 PHASE 2: STRING ARRAY BASELINE")
    string_schema = debugger.get_string_array_schema()
    
    await debugger.test_schema(
        string_schema,
        "gpt-4o",
        simple_prompt, 
        "String Array Schema (Control)"
    )
    
    # Phase 3: Full rich schema test
    print("\n🧪 PHASE 3: FULL RICH SCHEMA")
    rich_schema = debugger.get_rich_attachment_schema()
    
    await debugger.test_schema(
        rich_schema,
        "gpt-4o", 
        rich_prompt,
        "Full Rich Attachment Schema"
    )
    
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE")
    print("Check the results above to identify where object schema enforcement fails.")

if __name__ == "__main__":
    asyncio.run(main())