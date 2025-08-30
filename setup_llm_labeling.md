# 🤖 LLM Labeling Setup Guide

## Current Status
✅ **System Ready**: All LLM labeling components implemented and tested
⚠️ **API Key Needed**: Requires OpenAI or Anthropic API key to proceed

## Quick Setup

### 1. Set API Key
```bash
# For OpenAI (recommended - cheaper with gpt-4o-mini)
export OPENAI_API_KEY='your-openai-api-key'

# OR for Anthropic
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

### 2. Generate LLM Labels

#### Test Run (5 examples, ~$0.02)
```bash
python generate_llm_labels.py \
  --input data/hrm_dlp_final/train_labeled.jsonl \
  --max_examples 5 \
  --provider openai \
  --model gpt-4o-mini
```

#### Full Training Set (1,854 examples, estimated ~$7-15)
```bash
python generate_llm_labels.py \
  --input data/hrm_dlp_final/train_labeled.jsonl \
  --provider openai \
  --model gpt-4o-mini \
  --output data/hrm_dlp_final/train_llm_labeled.jsonl
```

#### Generate All Splits
```bash
# Training data
python generate_llm_labels.py \
  --input data/hrm_dlp_final/train_labeled.jsonl \
  --output data/hrm_dlp_final/train_llm_labeled.jsonl \
  --provider openai --model gpt-4o-mini

# Validation data  
python generate_llm_labels.py \
  --input data/hrm_dlp_final/val_labeled.jsonl \
  --output data/hrm_dlp_final/val_llm_labeled.jsonl \
  --provider openai --model gpt-4o-mini

# Test data
python generate_llm_labels.py \
  --input data/hrm_dlp_final/test_labeled.jsonl \
  --output data/hrm_dlp_final/test_llm_labeled.jsonl \
  --provider openai --model gpt-4o-mini
```

### 3. Compare Label Quality
```bash
python compare_labeling_approaches.py \
  --rule_file data/hrm_dlp_final/train_labeled.jsonl \
  --llm_file data/hrm_dlp_final/train_llm_labeled.jsonl \
  --output comparison_report.json
```

### 4. Update Training Data
```bash
# Copy LLM-labeled files to replace rule-based ones
cp data/hrm_dlp_final/train_llm_labeled.jsonl data/hrm_dlp_final/train_labeled.jsonl
cp data/hrm_dlp_final/val_llm_labeled.jsonl data/hrm_dlp_final/val_labeled.jsonl
cp data/hrm_dlp_final/test_llm_labeled.jsonl data/hrm_dlp_final/test_labeled.jsonl
```

### 5. Retrain Model
```bash
./retrain_with_labels.sh
```

## Expected Improvements

Based on training analysis showing high validation loss (6.45 vs 0.37 training), LLM labels should provide:

✅ **Better Generalization**: Lower validation/training loss gap
✅ **Contextual Understanding**: Business appropriateness vs pattern matching  
✅ **Improved Discrimination**: Wider risk score ranges for better model training
✅ **Strategic Alignment**: Complements regex systems instead of competing

## Cost Estimates

- **Test (5 examples)**: ~$0.02
- **Training set (1,854)**: ~$7-15  
- **All splits (2,653)**: ~$10-20
- **Total project cost**: ~$15-25

## Monitoring Progress

The system provides real-time progress updates:
- Success rate tracking
- Cost accumulation  
- Response time monitoring
- Error logging with recovery

## Quality Assurance

Each LLM response includes:
- Structured JSON labels (sensitivity, exposure, context, obfuscation)
- Detailed reasoning for each dimension
- Business context analysis
- Confidence validation