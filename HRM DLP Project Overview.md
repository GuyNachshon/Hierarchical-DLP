# HRM‑First DLP — Project Overview

This document gives a single‑page(ish) view of the project: vision, scope, architecture, deliverables, timelines, and who owns what. It ties together the two deep‑dive docs on the canvas:

- **HRM → DLP: Model Diff & Train Plan** (model architecture, heads, losses, export, serving)
- **HRM DLP Data Spec & Generator** (dataset schema, DSL, weak labels, augmentation)

---

## 1) Vision & Value

**Problem:** Legacy DLP over‑blocks and misses context because decisions are rule/regex‑driven.

**Our answer:** A native, **HRM‑first** DLP that reasons about **who** is sending **what** to **whom**, within **which workflow**—and explains every decision. We combine:

- **HRM core** (small hierarchical model) → content understanding + spans + conversation memory
- **TrustScore** (behavioral recipient trust) → no manual allowlists
- **Neural/Learned Bloom** → microsecond “reflexes” for trust/novelty/evidence caching
- **Probabilistic Risk Fusion + Policy DSL** → fewer FPs at the same recall, with explainability

**Target outcomes:**

- **≤ 300 ms** verdict for a typical 2 KB message
- **≥ 25–40% FP reduction @ 95% recall** vs regex+small‑transformer baseline
- Every action has **evidence** (spans + rationale) and is **privacy‑safe** by default

---

## 2) Scope (MVP)

- **Channels:** Email (web), Chat (basic), PR/Code (stretch)
- **Decisions:** Allow / Warn / Block
- **Evidence:** Token spans (PII, secrets, NDA/Matter) + rationale chips
- **Deployment:** Browser extension + local agent OR endpoint agent
- **Data:** Synthetic + weak labels; small distillation pass for nuance

Out of scope for MVP: OCR, deep file parsing, server‑side mail proxy, mobile apps.

---

## 3) System Architecture (bird’s‑eye)

```
[Capture/Agent] → [Serializer] →  HRM Core  →  [Conversation Memory]
                               ↓                 ↑
                      [Spans + Doc Scores]       │
                               ↓                 │
                            [TrustScore]         │
                               ↓                 │
                           [Risk Fusion]         │
                               ↓                 │
                          [Policy Engine]  →  [Decision + Evidence]
                               ↓
                           [Audit Store]
```

**Key flows**

- **Compose‑time gating:** extension/agent serializes content → HRM runs → TrustScore lookup → Risk Fusion → Policy → banner
- **Post‑send updates:** memory summary write; TrustScore online update; audit log

---

## 4) Components (what each does)

- **Serializer (DSL):** Packs role, recipients, thread, subject, body, attachments, links, metadata into a single structured string; maintains span offset map.
- **HRM Core:** 25–35M params, two‑speed encoder (fast/slow) with heads for 4 doc scores + BIO spans + 256‑D memory summary. Fixed compute (ACT off in v1). ONNX int8 for edge.
- **Conversation Memory:** Local KV keyed by thread hash; stores summary vector + flags (no raw text) with TTL.
- **TrustScore Engine:** Behavioral recipient/domain trust from historical interactions (frequency/recency/diversity/context quality/reputation). Logistic/GBDT; online updates.
- **Neural/Learned Bloom:**
  - Sandwiched LBF for fast “likely‑trusted” prior on recipients
  - Bloom/Bloomier for re‑encounter suppression of previously adjudicated benign secrets/PII
  - Bloom sidecar for quick “NDA/Matter established?” lookups
- **Risk Fusion:** Calibrated blend of HRM scores and TrustScore into `FinalRisk`.
- **Policy DSL:** Declarative thresholds/guards/overrides (probabilistic, explainable). Emits Allow/Warn/Block + rationale.
- **UI (Extension/Agent):** Non‑blocking banner with chips (Sensitivity / Exposure / Context / Trust), decision, and “Show evidence” highlights.
- **Audit Store:** Privacy‑safe logs (hashes, offsets, scores, rationale, model/policy versions).

---

## 5) Interfaces (summary)

- **/scan** → input: serialized content & metadata; output: scores, spans, memory flags, decision, rationale.
- **/memory/get|update** → read/write thread summaries.
- **/trust/get|event** → TrustScore and feature‑level introspection; online event updates.
- **/policy/validate|apply** → dry‑run & apply new policies with CI.

(Full details live in the two deep‑dive docs.)

---

## 6) Data & Modeling Plan (how we train)

- **Dataset**: see **HRM DLP Data Spec & Generator**
  - 60k synthetic train + 5k dev + 5k test; stratified by role/domain/span types
  - Weak labels for doc heads; regex/dicts (with Luhn/entropy) for spans
  - Augment: obfuscations, lookalikes, legal‑to‑personal hard negatives
- **Training**: see **HRM → DLP: Model Diff & Train Plan**
  - Loss = BCE(doc) + CE(BIO) + 0.3*mask‑denoise + 0.2*section‑shuffle
  - 1‑step gradient over fixed segments; bf16; early stop on FP\@95R
  - Temperature scaling per head; ONNX (unrolled 1024) + int8

---

## 7) Success Metrics (what we measure)

- **Detection quality:** AUPRC per head; **FP rate @ 95% recall** (primary)
- **Evidence quality:** Span F1 (macro), Precision\@k for high‑severity spans
- **Stability:** decision consistency under ±200‑token window shift
- **UX:** override rate, time‑to‑send, banner dwell time
- **Perf:** p95 inference latency per 2 KB window; cold start; memory fetch time

---

## 8) Timeline (MVP: \~4 weeks)

**48‑hour starter**

- Data generator + weak labelers; tokenization; model scaffold; quick train
- Minimal TrustScore (recency/frequency); Scan API; simple Policy; extension banner

**Week 1**

- Obfuscation head; conversation memory; LBF v0; calibration; perf tuning

**Week 2**

- Teacher distillation on tricky cases; TrustScore v1 (full features); Policy CI; audit store

**Week 3**

- Robustness sweeps (homoglyph/base64/lookalikes); threshold tuning per tenant; telemetry

**Week 4**

- Tenantization & packaging; A/B vs tiny transformer; ship internal pilot

---

## 9) Risks & Mitigations

- **Export friction (slow tier)** → unroll for ONNX; keep TorchScript fallback
- **Weak‑label noise** → hard negatives + multi‑task aux losses + small distillation set
- **Latency variance** → early‑exit when decisive; int8 quant; cap windows
- **Cold‑start trust** → conservative defaults + LBF prior + WHOIS/MX heuristics
- **Explainability gaps** → span head + rationale templates; store thresholds and versions with each verdict

---

## 10) Ownership & Working Agreements

- **ML**: HRM model, calibration, export, metrics, A/B baseline
- **Backend**: Scan/Trust/Memory APIs, Policy DSL, audit store, CI
- **Endpoint/Extension**: capture, UI, latency, packaging
- **PM/UX**: policy semantics, evidence UX, demo scripting, KPIs

**Cadence:** daily 15‑min standup; twice‑weekly demo; metrics reviewed weekly (FP\@95R, p95 latency, override rate).

---

## 11) Demo Storyline (Portugal HQ)

- **Case A** (Legal → established counsel): Sensitivity↑, Context↑, Trust↑ → **Allow with warning**; highlights show PAN + NDA/Matter.
- **Case B** (Marketing → proton): Sensitivity↑, Exposure↑, Trust↓ → **Block**; highlights show PAN + “untrusted recipient”.
- **Before/After** toggle: regex DLP blocks both vs HRM DLP only blocks risk.

---

## 12) Open Questions

- How “strict” should the default policy be for cold‑start tenants?
- Do we ship ACT (adaptive compute) early for hard cases or keep deterministic latency for v1?
- Tenant‑side storage: SQLite vs embedded KV; rotation policy for memory/Bloom salts.

---

## 13) Repo Skeleton (guidance)

```
hrm-dlp/
  configs/        # base.yaml, train.yaml, export.yaml
  hrm_dlp/        # tokenizer, dsl, model, heads, losses, train, eval, export, serve
  data/           # jsonl + spm model
  policies/       # policy DSL + tests
  server/         # FastAPI app + Docker
  extension/      # MV3 extension
  tests/          # unit + E2E
  README.md
```

This overview is the north star. Use it alongside the two deep‑dives to implement, measure, and ship the MVP.

