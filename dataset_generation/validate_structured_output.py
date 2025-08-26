"""
Quick validator for OpenAI structured outputs (strict JSON schema).

Usage examples:
  uv run python HRM/scripts/validate_structured_output.py \
    --model gpt-4o --subject "Quarterly update" --body "Summarize Q2 KPIs" \
    --recipient you@example.com

Exits with code 0 on success; prints server errors on failure.
"""

import os
import sys
import json
import argparse

try:
    import openai  # type: ignore
except Exception:
    print("openai package not installed. Try: uv add openai or pip install openai")
    sys.exit(2)


def build_response_format(strict: bool = True) -> dict:
    """Schema aligned with generator batch usage (strict mode requirements)."""
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "dlp_example",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "channel": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "recipients": {"type": "array", "items": {"type": "string"}},
                    "attachments": {"type": "array", "items": {"type": "string"}},
                    "links": {"type": "array", "items": {"type": "string"}},
                },
                # In strict mode, required must include every key listed in properties.
                "required": [
                    "channel",
                    "subject",
                    "body",
                    "recipients",
                    "attachments",
                    "links",
                ],
            },
        },
    }
    if strict:
        rf["json_schema"]["strict"] = True
    return rf


def main():
    ap = argparse.ArgumentParser(description="Validate OpenAI strict JSON schema with one request")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    ap.add_argument("--system", default="You are an assistant that returns only valid JSON matching the schema.")
    ap.add_argument("--subject", default="Status update")
    ap.add_argument("--body", default="Provide a brief status update about Q2 goals.")
    ap.add_argument("--recipient", action="append", default=["dlp@example.com"], help="Add a recipient (can repeat)")
    ap.add_argument("--channel", default="email")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--no-strict", action="store_true", help="Disable strict mode (diagnostics)")
    args = ap.parse_args()

    if not args.api_key:
        print("OPENAI_API_KEY not set. Provide --api-key or set env.")
        sys.exit(2)

    client = openai.OpenAI(api_key=args.api_key)

    rf = build_response_format(strict=not args.no_strict)

    system = args.system + "\nRespond ONLY with valid JSON per the schema. No extra text."
    user = (
        f"Generate a single {args.channel} message as JSON.\n"
        f"Subject hint: {args.subject}\nBody hint: {args.body}\n"
        f"Recipients must include: {args.recipient}"
    )

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            response_format=rf,
        )
    except Exception as e:
        print("Request failed:")
        print(e)
        sys.exit(1)

    content = resp.choices[0].message.content
    print("Raw content:\n", content)
    try:
        data = json.loads(content)
        print("\nParsed JSON (keys):", list(data.keys()))
        # Quick shape check
        missing = [k for k in rf["json_schema"]["schema"]["required"] if k not in data]
        if missing:
            print("\nWARNING: Missing required keys in response:", missing)
            sys.exit(3)
    except Exception as je:
        print("\nFailed to parse JSON:", je)
        sys.exit(4)

    print("\nOK: Structured output matches schema.")


if __name__ == "__main__":
    main()

