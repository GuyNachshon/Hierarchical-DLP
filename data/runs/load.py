import os
from pathlib import Path
import json

from datasets import Dataset, DatasetDict


def _to_list_of_str(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, dict):
        # Preserve object by JSON-stringifying
        return [json.dumps(value, ensure_ascii=False)]
    return [str(value)]


def normalize_record(rec: dict) -> dict:
    return {
        "channel": str(rec.get("channel", "email")),
        "user": rec.get("user", {}) if isinstance(rec.get("user"), dict) else {},
        "recipients": [str(r) for r in (rec.get("recipients") or [])],
        "subject": str(rec.get("subject", "")),
        "body": str(rec.get("body", "")),
        "attachments": _to_list_of_str(rec.get("attachments", [])),
        "links": _to_list_of_str(rec.get("links", [])),
        "thread": rec.get("thread") if isinstance(rec.get("thread"), dict) else None,
        "labels": rec.get("labels") if isinstance(rec.get("labels"), dict) else None,
        "spans": rec.get("spans") if isinstance(rec.get("spans"), list) else None,
        "_metadata": rec.get("_metadata") if isinstance(rec.get("_metadata"), dict) else {},
    }


def load_normalized_jsonl_rows(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Skip malformed lines
                continue
            rows.append(normalize_record(obj))
    return rows


_dir = Path("run_20250824_123640_a2f52bf9/split_outputs")

test_file = _dir / "test_examples_augmented.jsonl"
train_file = _dir / "train_examples_augmented.jsonl"
val_file = _dir / "val_examples_augmented.jsonl"

test_rows = load_normalized_jsonl_rows(test_file)
train_rows = load_normalized_jsonl_rows(train_file)
val_rows = load_normalized_jsonl_rows(val_file)

# unify schema keys across splits before creating datasets
user_keys = set()
for recs in (train_rows, val_rows, test_rows):
    for r in recs:
        user_keys.update((r.get("user") or {}).keys())
meta_keys = set()
for recs in (train_rows, val_rows, test_rows):
    for r in recs:
        meta_keys.update((r.get("_metadata") or {}).keys())

def fill_keys(rows):
    for r in rows:
        u = r.get("user") or {}
        r["user"] = {k: str(u.get(k, "")) for k in user_keys}
        m = r.get("_metadata") or {}
        r["_metadata"] = {
            k: (json.dumps(m.get(k)) if isinstance(m.get(k), (dict, list)) else str(m.get(k, "")))
            for k in meta_keys
        }

fill_keys(train_rows)
fill_keys(val_rows)
fill_keys(test_rows)

train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)
test_ds = Dataset.from_list(test_rows)

ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

# Push to Hub (ensure token is set appropriately)
ds.push_to_hub("guychuk/synthetic-agentic-dlp", token=os.getenv("HUGGINGFACE_API_TOKEN"))
