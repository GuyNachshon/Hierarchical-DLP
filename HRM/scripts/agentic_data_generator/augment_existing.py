"""
Augment an existing run's generated examples without re-generating the base set.

Usage:
  uv run python HRM/scripts/agentic_data_generator/augment_existing.py \
    --session-dir data/runs/<run_id> [--ratio 0.3] [--target flip|low_risk|high_risk]

Writes augmented examples alongside the originals under split_outputs/*_examples_augmented.jsonl
and also emits augmented_<split>.jsonl into the original run's output_dir if found.
"""

import argparse
import asyncio
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from config import AgenticConfig
from agents.augmentation_agent import AugmentationAgent
from agents.base_agent import GeneratedExample
from batch.fixed_batch_processor import FixedBatchProcessor


def _load_examples(run_dir: Path, split: str) -> List[Dict]:
    f = run_dir / "split_outputs" / f"{split}_examples.jsonl"
    if not f.exists():
        return []
    out = []
    with f.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _save_augmented(run_dir: Path, split: str, augmented: List[Dict]):
    out_dir = run_dir / "split_outputs"
    out_dir.mkdir(exist_ok=True)
    f = out_dir / f"{split}_examples_augmented.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for ex in augmented:
            fh.write(json.dumps(ex) + "\n")


async def _augment_split(agent: AugmentationAgent, examples: List[Dict], ratio: float, target: str) -> List[Dict]:
    if not examples:
        return []
    n = max(1, int(len(examples) * ratio))
    subset = examples[:n]

    def _to_list_of_str(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, dict):
            # Preserve object by JSON-stringifying
            return [json.dumps(value, ensure_ascii=False)]
        return [str(value)]

    async def flip_one(ex: Dict) -> Optional[Dict]:
        try:
            ge = GeneratedExample(
                channel=ex.get('channel', 'email'),
                user=ex.get('user', {}),
                recipients=ex.get('recipients', []),
                subject=ex.get('subject', ''),
                body=ex.get('body', ''),
                attachments=ex.get('attachments', []),
                links=ex.get('links', []),
                thread=ex.get('thread'),
                labels=ex.get('labels'),
                spans=ex.get('spans'),
                meta=ex.get('_metadata', {})
            )
            if target == 'flip':
                current = agent._infer_current_risk_level(ge)
                goal = 'low_risk' if current in ('medium_risk', 'high_risk') else 'high_risk'
            else:
                goal = target
            flipped = await agent.flip_scenario(ge, goal)
            if not flipped:
                return None
            out = {
                'channel': str(flipped.channel or 'email'),
                'user': flipped.user if isinstance(flipped.user, dict) else {},
                'recipients': [str(r) for r in (flipped.recipients or [])],
                'subject': str(flipped.subject or ''),
                'body': str(flipped.body or ''),
                'attachments': _to_list_of_str(flipped.attachments),
                'links': _to_list_of_str(flipped.links),
                'thread': flipped.thread if isinstance(flipped.thread, dict) else None,
                'labels': flipped.labels if isinstance(flipped.labels, dict) else None,
                'spans': flipped.spans if isinstance(flipped.spans, list) else None,
                '_metadata': {**ex.get('_metadata', {}), 'augmented': True, **(flipped.meta or {})}
            }
            return out
        except Exception as e:
            print(f"Failed to augment one example: {e}")
            return None

    tasks = [flip_one(ex) for ex in subset]
    done = await asyncio.gather(*tasks)
    return [d for d in done if d]


async def main():
    ap = argparse.ArgumentParser(description="Augment an existing run with flipped scenarios")
    ap.add_argument("--session-dir", required=True, help="Path to run directory (data/runs/<run_id>)")
    ap.add_argument("--ratio", type=float, default=0.3, help="Portion of examples to augment per split")
    ap.add_argument("--target", choices=["flip", "low_risk", "high_risk"], default="flip")
    ap.add_argument("--openai-key", default=None)
    ap.add_argument("--anthropic-key", default=None)
    ap.add_argument("--use-batch", action="store_true", help="Use batch/concurrent processor for augmentation")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="Preferred provider for batching")
    ap.add_argument("--model", default=None, help="Override model name for provider (e.g., claude-3-haiku-20240307)")
    ap.add_argument("--batch-threshold", type=int, default=20, help="Min requests to use batch API (else concurrent)")
    args = ap.parse_args()

    run_dir = Path(args.session_dir)
    assert run_dir.exists(), f"Session directory not found: {run_dir}"

    # Minimal config just for client init
    cfg = AgenticConfig(enable_batch_api=args.use_batch, batch_threshold=args.batch_threshold)
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key
    agent = AugmentationAgent(cfg)
    if not getattr(agent, 'clients', None):
        print("❌ No LLM clients available. Set OPENAI_API_KEY/ANTHROPIC_API_KEY or pass --openai-key/--anthropic-key.")
        return

    splits = ["train", "val", "test"]
    totals = {}
    for split in splits:
        base = _load_examples(run_dir, split)
        if not base:
            print(f"{split}: no base examples found; skipping")
            continue
        if args.use_batch:
            # Build (system, user) pairs via augmentation agent
            n = max(1, int(len(base) * args.ratio))
            subset = base[:n]
            pairs = []
            for ex in subset:
                ge = GeneratedExample(
                    channel=ex.get('channel', 'email'), user=ex.get('user', {}),
                    recipients=ex.get('recipients', []), subject=ex.get('subject', ''),
                    body=ex.get('body', ''), attachments=ex.get('attachments', []),
                    links=ex.get('links', []), thread=ex.get('thread'), labels=ex.get('labels'),
                    spans=ex.get('spans'), meta=ex.get('_metadata', {})
                )
                if args.target == 'flip':
                    cur = agent._infer_current_risk_level(ge)
                    goal = 'low_risk' if cur in ('medium_risk', 'high_risk') else 'high_risk'
                else:
                    goal = args.target
                pp = agent.build_flip_prompts(ge, goal)
                pairs.append((pp['system'], pp['user']))

            def _to_list_of_str(value) -> list:
                if value is None:
                    return []
                if isinstance(value, list):
                    return [str(v) for v in value if v is not None]
                if isinstance(value, dict):
                    return [json.dumps(value, ensure_ascii=False)]
                return [str(value)]

            processor = FixedBatchProcessor(cfg)
            # Choose model with provider preference
            provider, model = (args.provider, args.model) if args.model else processor.client_manager.choose_model(args.provider)
            results = await processor.process_requests_with_fixed_model(pairs, provider, model)
            # Parse
            aug = []
            for ex, raw in zip(subset, results):
                if not raw:
                    continue
                data = agent._extract_json_from_response(raw)
                if not data:
                    continue
                ge = GeneratedExample(
                    channel=ex.get('channel', 'email'), user=ex.get('user', {}),
                    recipients=ex.get('recipients', []), subject=ex.get('subject', ''),
                    body=ex.get('body', ''), attachments=ex.get('attachments', []),
                    links=ex.get('links', []), thread=ex.get('thread'), labels=ex.get('labels'),
                    spans=ex.get('spans'), meta=ex.get('_metadata', {})
                )
                if not agent._validate_flip_transformation(ge, data):
                    continue
                out = {
                    'channel': str(data.get('channel', 'email')),
                    'user': data.get('user', {}) if isinstance(data.get('user'), dict) else {},
                    'recipients': [str(r) for r in (data.get('recipients') or [])],
                    'subject': str(data.get('subject', '')),
                    'body': str(data.get('body', '')),
                    'attachments': _to_list_of_str(data.get('attachments', [])),
                    'links': _to_list_of_str(data.get('links', [])),
                    'thread': data.get('thread') if isinstance(data.get('thread'), dict) else None,
                    'labels': data.get('labels') if isinstance(data.get('labels'), dict) else None,
                    'spans': data.get('spans') if isinstance(data.get('spans'), list) else None,
                    '_metadata': {**ex.get('_metadata', {}), 'augmented': True, 'agent': 'augmentation'}
                }
                aug.append(out)
        else:
            aug = await _augment_split(agent, base, args.ratio, args.target)
        _save_augmented(run_dir, split, aug)
        totals[split] = (len(base), len(aug))
        print(f"{split}: augmented {len(aug)}/{len(base)}")

    # If the run recorded an output_dir, also write augmented files there
    session_file = run_dir / "session.json"
    if session_file.exists():
        try:
            info = json.loads(session_file.read_text())
            output_dir = Path(info.get("output_dir", ""))
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                for split in totals:
                    aug_file = run_dir / "split_outputs" / f"{split}_examples_augmented.jsonl"
                    if aug_file.exists():
                        out_file = output_dir / f"augmented_{split}.jsonl"
                        out_file.write_text(aug_file.read_text())
                        print(f"wrote {out_file}")
        except Exception:
            pass

    print("Done:", totals)


async def get_anthropic_batches(client):
    batches = client.messages.batches.list(limit=20)
    # keep only completed batches from today
    today_date = date.today()
    batches = list(map(lambda x: x.id, filter(lambda x: x.created_at.date() == today_date, batches.data)))
    for batch_id in batches:
        batch_url = client.messages.batches.retrieve(
            batch_id,
        ).results_url

        res = requests.get(batch_url, headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
        res = res.text
        with open(f"{batch_id}.jsonl", "w+") as f:
            f.write(res)


if __name__ == "__main__":
    asyncio.run(main())
