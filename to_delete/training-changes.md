# HRM → DLP: Model Diff & Train Plan

This doc lists **precise, code‑level changes** to adapt the sapientinc/HRM repository and paper design into a production‑oriented **DLP classifier+tagger** with conversation memory. It includes file‑by‑file diffs, model parameters, training objectives, export, and serving.

---

## 1) Goals & Deliverables

- **Inputs:** structured DSL string (email/chat/PR/upload) with role, recipients, subject, body, attachments, links, metadata.
- **Outputs:**
  - Doc‑level scores: `Sensitivity`, `ExposureRisk`, `ContextConsistency`, `ObfuscationRisk` (sigmoid, calibrated).
  - Token BIO tags: `EMAIL, PHONE, PAN, SSN_LIKE, SECRET_KEY, DB_URI, NDA_TERM, MATTER_ID, NAME, ADDRESS, O`.
  - 256‑D **memory summary** (+ boolean flags).
- **Training:** multi‑task (doc BCE + BIO CE + aux mask‑denoise + aux section‑shuffle), **1‑step gradient** w/ deep supervision over fixed segments.
- **Inference:** fixed compute budget (deterministic), ONNX export (unrolled), int8 quant, <300 ms verdict for \~2 KB.

---

## 2) High‑Level Architectural Changes

| Component    | Upstream HRM                    | DLP Modifications                                                                    |
| ------------ | ------------------------------- | ------------------------------------------------------------------------------------ |
| Input stream | Puzzle tokens                   | **Typed DSL** serialization of email/chat, etc.                                      |
| Fast module  | Transformer encoder blocks      | Keep; d=384, L=8, H=6; RoPE, GLU, RMSNorm.                                           |
| Slow module  | Recurrent encoder over segments | Keep; **stride=64 tokens**, 2 blocks; **learned gates** to fuse {input, fast, slow}. |
| State init   | Fixed/random                    | **Learned slow seed**; optional **thread memory seed**.                              |
| ACT/Halting  | Optional, Q‑learning            | **Disable initially** for deterministic latency; can add later.                      |
| Heads        | Task decoder                    | **Doc score head (4 logits)**, **Span BIO head**, **Memory summary head**.           |
| Loss         | Seq CE                          | **BCE(doc)+CE(BIO)+aux** (mask‑denoise, shuffle).                                    |
| Export       | Research loop                   | **Unrolled ONNX (1024)**, int8; TorchScript fallback.                                |

---

## 3) Model Config (YAML)

```yaml
# configs/base.yaml
vocab_size: 16000
max_len: 1024
# Fast tier
d_model: 384
n_fast_layers: 8
n_heads: 6
mlp_mult: 4
rope_theta: 10000
# Slow tier
n_slow_blocks: 2
slow_update_stride: 64    # update slow state every 64 tokens
slow_hidden_dim: 384
slow_gate: true           # enable learned fusion gates
# Heads
score_heads: ["sensitivity","exposure","context","obfuscation"]
span_label_map: ["B-EMAIL","I-EMAIL","B-PHONE","I-PHONE","B-PAN","I-PAN","B-SSN","I-SSN","B-SECRET","I-SECRET","B-DBURI","I-DBURI","B-NDA","I-NDA","B-MATTER","I-MATTER","B-NAME","I-NAME","B-ADDR","I-ADDR","O"]
memory_vec_dim: 256
# Training
mask_denoise_prob: 0.15
shuffle_prob: 0.10
segments: 2               # fixed segments for 1-step gradient
segment_len: 64           # 2*64*stride coverage via pooling
```

---

## 4) File‑by‑File Changes

### 4.1 `tokenizer/` → add SentencePiece wrapper

- New: `hrm_dlp/tokenizer.py`
  - Train SPM (16k) on DSL text; expose `encode(text, max_len)` and `decode(ids)`.

### 4.2 `data/` → serializers & loaders

- New: `hrm_dlp/dsl.py`
  - `serialize(example) -> str` from schema.
  - `align_spans(raw_body, serialized_text) -> BIO tag ids` (offset mapping tests included).
- New: PyTorch `Dataset` reading JSONL, returning `{ids, attn_mask, doc_labels[4], bio_tags[T], mem_seed[256]}`.

### 4.3 `model/` → HRM encoder with gated fusion

- Modify upstream `FastBlock` to ensure RoPE+RMSNorm+GLU are available (mirror upstream choices).
- New: `SlowRecurrentBlock`:
  - For segments of `slow_update_stride`, compute pooled fast features (mean + max + CLS) → `u_t`.
  - Update slow state with GRU‑like cell: `s_t = GRU(s_{t-1}, u_t)`.
  - **Fusion gate** inside fast path: for each token `x`, previous slow `s_cur` and local fast `h`: `g = σ(W_g [x||h||s_cur])`; `h' = g ⊙ Ux + (1-g) ⊙ Uh`.
- New heads in `heads.py`:
  - `DocScoreHead(d_model, 4)` → logits.
  - `SpanTagHead(d_model, n_tags)` → per‑token logits.
  - `MemorySummaryHead(d_model, 256)`.
- Entry module `HRMDLP.forward(ids, mask, slow_seed=None)` returns `(doc_logits[4], span_logits[T,n_tags], mem_vec[256])`.

### 4.4 `training/` → objectives & 1‑step gradient

- New: `losses.py`:
  - `bce_logits(doc_logits, y)`, `ce_bio(span_logits, tags)`.
  - Aux `mask_denoise_loss` (perturb inputs) and `shuffle_detect_loss` (binary head optional; or classify positional token).
- New: `train.py` loop:
  - **Fixed segments**: split sequence into `segments` parts; **detach slow state** between parts to mimic 1‑step gradient stability.
  - Deep supervision: compute doc+span losses per segment; average.
  - Mixed precision (bf16), grad accumulation by token budget.

### 4.5 `calibration/`

- New: temperature scaling per doc head; save `calib.json`.

### 4.6 `export/`

- New: `export_onnx.py` unrolls slow updates for `max_len=1024`, `stride=64`, `segments=2`.
- Quantize dynamic int8 with onnxruntime‑tools; parity check script reports ΔAUPRC.

### 4.7 `serve/`

- New: `serve_fastapi.py` implementing `/scan` (tokenize → infer → calibrate → output spans/scores/memory flags). Optional `/warmup`.

---

## 5) Training Setup

**Hyperparameters**

```
optimizer: AdamW (β=0.9,0.95)  lr=3e-4  weight_decay=0.02
sched: cosine with 3k warmup steps
mixed_precision: bf16
batch_tokens: ~2–3M via grad accumulation
epochs: 2 (start), early stop on dev FP@95R
label_smoothing: 0.05 (doc), 0.0 (BIO)
```

**Loss**

```
L = BCE(doc) + CE(BIO) + 0.3*MaskDenoise + 0.2*SectionShuffle
```

**Curriculum**

1. Epoch 0: window 512, easy samples, no obfuscations.
2. Epoch 1+: window 1024, add hard negatives + obfuscations.

**Dev metrics**

- Doc: AUPRC per head; **FP\@95% recall** (primary); ECE calibration.
- Span: F1 (macro) + Precision\@k for high‑severity spans.
- Stability: decision consistency under ±200‑token shift.

---

## 6) Inference & Early Exit

- Fixed compute: segments=2, stride=64.
- After first segment, if `Sensitivity < 0.2` and `Exposure < 0.2` → **allow early**.
- If `Sensitivity > 0.9` and `(Exposure > 0.7 or Obfuscation > 0.6)` → **block early** (policy may still adjust).

---

## 7) Export, Quant, Serving

1. Export ONNX (opset 17), no dynamic axes; validate on dev.
2. Quantize int8 dynamic; re‑score dev.
3. Serve with onnxruntime‑gpu (server) or ORT‑web (edge) later.

**/scan response**

```
{
  "scores": {"sensitivity":0.91,"exposure":0.24,"context":0.78,"obfuscation":0.06},
  "spans": [{"type":"PAN","start":128,"end":147}, {"type":"NDA","start":10,"end":13}],
  "memory": {"vec": "...base64...", "flags": {"has_nda": true}},
  "decision": "ALLOW_WITH_WARNING",
  "rationale": ["PII detected","High legal context"]
}
```

---

## 8) Unit Tests & Validation

- **Span alignment**: round‑trip DSL serializer preserves body offsets; BIO tags line up.
- **State unroll parity**: Torch vs ONNX outputs within tolerance.
- **Gate ablation**: off vs on → expect +AUPRC for exposure/context.
- **Latency budget**: p95 < 200 ms for 2 KB window on target box.

---

## 9) Roadmap (model‑side)

- v1: deterministic compute, ACT disabled.
- v1.1: optional ACT halting for “hard” cases (budgeted extra steps).
- v1.2: distillation from LLM rationales to refine spans & context.
- v2: multimodal (image/PDF OCR tokens), retrieval‑augmented memory.

