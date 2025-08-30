#!/bin/bash

echo "🤖 Full LLM Labeling Pipeline"
echo "============================="

# Check for API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ No API keys found!"
    echo "Please set: export OPENAI_API_KEY='your-key' or ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Choose provider and model
if [ ! -z "$OPENAI_API_KEY" ]; then
    PROVIDER="openai"
    MODEL="gpt-4o-mini"
    echo "✅ Using OpenAI with $MODEL"
else
    PROVIDER="anthropic" 
    MODEL="claude-3-haiku-20240307"
    echo "✅ Using Anthropic with $MODEL"
fi

echo ""
echo "📊 Current Rule-Based Labels Analysis:"
python -c "
import json
import numpy as np

def analyze_labels(file_path, label_key='labels'):
    labels_data = {'sensitivity': [], 'exposure': [], 'context': [], 'obfuscation': []}
    with open(file_path, 'r') as f:
        for line in f:
            try:
                example = json.loads(line)
                if label_key in example:
                    for label_type, values_list in labels_data.items():
                        if label_type in example[label_key]:
                            values_list.append(example[label_key][label_type])
            except: continue
    
    for label_type, values in labels_data.items():
        if values:
            values = np.array(values)
            print(f'   {label_type.capitalize():<12}: mean={values.mean():.3f}, std={values.std():.3f}, range={values.max()-values.min():.3f}')

analyze_labels('data/hrm_dlp_final/train_labeled.jsonl')
"

echo ""
echo "🔄 Starting LLM Labeling Process..."
echo ""

# Label training data
echo "1️⃣ Labeling training data (1,854 examples)..."
python generate_llm_labels.py \
    --input data/hrm_dlp_final/train_labeled.jsonl \
    --output data/hrm_dlp_final/train_llm_labeled.jsonl \
    --provider $PROVIDER \
    --model $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Training data labeling failed!"
    exit 1
fi

# Label validation data  
echo ""
echo "2️⃣ Labeling validation data (400 examples)..."
python generate_llm_labels.py \
    --input data/hrm_dlp_final/val_labeled.jsonl \
    --output data/hrm_dlp_final/val_llm_labeled.jsonl \
    --provider $PROVIDER \
    --model $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Validation data labeling failed!"
    exit 1
fi

# Label test data
echo ""
echo "3️⃣ Labeling test data (399 examples)..."
python generate_llm_labels.py \
    --input data/hrm_dlp_final/test_labeled.jsonl \
    --output data/hrm_dlp_final/test_llm_labeled.jsonl \
    --provider $PROVIDER \
    --model $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Test data labeling failed!"
    exit 1
fi

echo ""
echo "📊 Comparing Label Quality..."
python compare_labeling_approaches.py \
    --rule_file data/hrm_dlp_final/train_labeled.jsonl \
    --llm_file data/hrm_dlp_final/train_llm_labeled.jsonl \
    --output llm_vs_rule_comparison.json

echo ""
echo "🔄 Backing up original rule-based files..."
mkdir -p data/hrm_dlp_final/rule_based_backup
cp data/hrm_dlp_final/train_labeled.jsonl data/hrm_dlp_final/rule_based_backup/
cp data/hrm_dlp_final/val_labeled.jsonl data/hrm_dlp_final/rule_based_backup/
cp data/hrm_dlp_final/test_labeled.jsonl data/hrm_dlp_final/rule_based_backup/

echo "✅ Rule-based labels backed up to data/hrm_dlp_final/rule_based_backup/"

echo ""
echo "🔄 Replacing training data with LLM labels..."
cp data/hrm_dlp_final/train_llm_labeled.jsonl data/hrm_dlp_final/train_labeled.jsonl
cp data/hrm_dlp_final/val_llm_labeled.jsonl data/hrm_dlp_final/val_labeled.jsonl  
cp data/hrm_dlp_final/test_llm_labeled.jsonl data/hrm_dlp_final/test_labeled.jsonl

echo "✅ Training data updated with LLM labels!"

echo ""
echo "🚀 Ready to retrain model!"
echo "Run: ./retrain_with_labels.sh"
echo ""
echo "📊 Expected improvements:"
echo "   ✅ Lower validation loss (better generalization)"
echo "   ✅ Better discrimination (wider risk score range)"  
echo "   ✅ Contextual understanding (business appropriateness)"
echo "   ✅ Strategic alignment (complements regex systems)"