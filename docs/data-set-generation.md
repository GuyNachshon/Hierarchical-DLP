# HRM DLP Data Spec & Generator

This doc defines the **dataset schema, DSL, label map, weak supervision rules, augmentation, and generator scripts** to produce training/dev/test corpora for the HRM‑first DLP.

---

## 1) JSONL Sample Schema (one row per example)

```json
{
  "channel": "email|chat|pr|upload",
  "user": {"role":"LEGAL","dept":"CORP","seniority":"SENIOR","id_hash":"U(...)"},
  "recipients": ["counsel@smith-legal.pt","cc@acme.com"],
  "thread": {"id_hash":"T(...)","age_days":7,"prior_msgs":3},
  "subject": "Portugal HQ – document packet",
  "body": "Per the NDA (Agreement #2145) and Matter M-4732...",
  "attachments": [{"name":"customers_q3.csv","mime":"text/csv","size":82000}],
  "links": ["https://share.acme.com/..."],
  "labels": {"sensitivity":1, "exposure":0, "context":1, "obfuscation":0},
  "spans": [
    {"type":"PAN","start":128,"end":147},
    {"type":"NDA_TERM","start":10,"end":13}
  ],
  "meta": {"base64":false,"homoglyph":false,"ts":"2025-08-15T12:03:00Z"}
}
```

---

## 2) DSL Serialization (input to model)

```
<CHANNEL email>
<USER role=LEGAL dept=CORP seniority=SENIOR>
<RECIPIENT primary=smith-legal.pt all=[smith-legal.pt,cc:acme.com]>
<THREAD id=T(...) age_days=7 prior_msgs=3>
<SUBJECT>Portugal HQ – document packet</SUBJECT>
<BODY>Per the NDA (Agreement #2145) and Matter M-4732...</BODY>
<ATTACHMENTS>customers_q3.csv|82KB;text/csv</ATTACHMENTS>
<LINKS>https://share.acme.com/...</LINKS>
<META base64=false homoglyph=false ts=2025-08-15T12:03Z>
```

- Serializer must keep a mapping from **raw body offsets → serialized offsets** to project BIO spans correctly.

---

## 3) Label Maps

### 3.1 Document Heads (4 logits)

- `sensitivity` ∈ {0,1} (presence of PII/PHI/IP/Secrets)
- `exposure` ∈ {0,1} (risk of exfiltration given recipients/tone)
- `context` ∈ {0,1} (conforms to legit workflow: NDA/Matter + role/channel alignment)
- `obfuscation` ∈ {0,1} (base64, zero‑width, homoglyphs, shorteners)

### 3.2 Token BIO Tags

```
B-EMAIL, I-EMAIL,
B-PHONE, I-PHONE,
B-PAN,   I-PAN,
B-SSN,   I-SSN,
B-SECRET,I-SECRET,   # keys, tokens, cert blocks
B-DBURI, I-DBURI,    # postgres://, mongodb://, etc.
B-NDA,   I-NDA,
B-MATTER,I-MATTER,
B-NAME,  I-NAME,
B-ADDR,  I-ADDR,
O
```

---

## 4) Weak Supervision Rules

### 4.1 Spans (regex/dicts)

- **EMAIL**: RFC‑lite regex.
- **PHONE**: E.164‑lite, allow separators.
- **PAN**: 13–19 digits with Luhn pass; mask negatives that fail Luhn.
- **SSN\_LIKE**: US/EU variants (high‑precision subset); add decoys for negatives.
- **SECRET\_KEY**: prefixes (`AKIA`, `SK_`, `xoxp-`, `ghp_`, `-----BEGIN` … `KEY-----`), JWT shape, high‑entropy strings.
- **DB\_URI**: `postgres://`, `mysql://`, `mongodb://`, `redis://`, `amqps://`.
- **NDA\_TERM**: \b(NDA|Non-Disclosure|Mutual NDA|Engagement|Agreement #\d+)\b
- **MATTER\_ID**: `[A-Z]-\d{3,5}` or `Matter(\s|#)?\w+` (tenant‑tuned).
- **NAME/ADDR**: gazetteers + simple NER (only keep high‑precision variants).

### 4.2 Document Heads (heuristics)

- `sensitivity=1` if any of {PAN, SECRET\_KEY, DB\_URI, SSN, NAME+EMAIL pair} present; else 0.
- `exposure=1` if any external personal domains in recipients (gmail/proton/outlook) **or** suspicious public links; else 0 when internal/trusted.
- `context=1` if (role∈{LEGAL,FINANCE} and NDA\_TERM present) **or** thread shows prior NDA flag; else 0.
- `obfuscation=1` if base64/zero‑width/homoglyph flags triggered.

---

## 5) Data Generation Pipeline

### 5.1 Templates

- **Roles**: LEGAL, FINANCE, HR, ENG, MARKETING, INTERN
- **Tones**: formal/legal, transactional, casual
- **Recipient buckets**: internal corp; sector‑looking (law/llp); personal webmail
- **Attachments**: csv/pdf/docx/zip names; sizes; optional header rows for csv
- **Bodies**: slot‑fill PII/keys/URIs; legal phrasing banks; matter/ticket IDs

### 5.2 Synthesis Steps

1. Sample role, recipients, thread meta.
2. Compose subject/body from tone + slots.
3. Inject spans per rules; build span list (start/end in body text).
4. Assign doc labels by heuristics.
5. Serialize to DSL; map spans to serialized offsets.
6. Emit JSONL line.

### 5.3 Quantities & Splits

- Train: 60k
- Dev:   5k
- Test:  5k
- Ensure **stratification** by role, domain type, and span types. Keep disjoint domain sets train/dev/test.

---

## 6) Augmentations & Hard Negatives

- **Obfuscations**: base64 chunks, homoglyphs, zero‑width joiners, URL shorteners.
- **Lookalikes**: `smith‑legal.pt` vs `sm1th‑legal.pt` (label exposure=1).
- **Legal‑to‑personal**: legal phrasing + personal domains (context=0, exposure=1).
- **Public docs with fake keys**: keys in README/license (sensitivity=0, spans may exist but suppressed after rules).

---

## 7) Generator Script Outline (`scripts/make_synth_data.py`)

- Reads YAML banks (`banks/phrases.yaml`, `banks/domains.yaml`, `banks/names.yaml`).
- Functions:
  - `make_recipients(bucket)` → list[str]
  - `make_body(role, tone, slots)` → str
  - `inject_spans(text, rules)` → spans[]
  - `assign_labels(spans, role, recipients, flags)` → doc labels
  - `serialize(example)` → DSL string + span offset map
  - write JSONL

CLI:

```
python scripts/make_synth_data.py \
  --out data/train.jsonl --n 60000 \
  --banks banks/ --seed 13
```

---

## 8) Validation Sets (hand‑crafted)

- **Nominal legal**: LEGAL→sector domain, NDA present; allowed.
- **Leak**: MARKETING→proton, PII and key; blocked.
- **Threaded**: message 1 establishes NDA; message 2 sends PII to same domain; allowed.
- **Obfuscation**: same as nominal but with base64 or homoglyphs; warn/block.

---

## 9) Data Quality Checks

- Span boundary tests (post‑serialization alignment).
- Luhn checks on PAN (positives/negatives ratio \~3:1).
- Domain bucket leakage: ensure personal domains not in train if tested for novelty.
- Label balance per head (target 45–55% where feasible; otherwise use class weights).

---

## 10) Delivery Artifacts

- `data/train.jsonl`, `dev.jsonl`, `test.jsonl`
- `data/spm/hrm_bpe.model` (SentencePiece)
- `banks/` YAMLs for phrases, domains, names, obfuscations
- `rules/` regex JSON for spans
- `reports/` data quality summary (counts by span/doc label, leakage checks)

---

## 11) Next Steps

- Add small **teacher‑distilled** set (2–5k) with rationales for tricky cases.
- Layer in limited **real corpora** (internal templates, redacted emails) with human curation.
- Extend to **PR/code** channel with code‑specific spans (tokens, config secrets).

