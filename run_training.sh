#!/bin/bash
# HRM-DLP Training Launch Script

echo "🚀 Starting HRM-DLP Training"
echo "=============================="

# Check if data exists
if [ ! -f "data/hrm_dlp_final/train.jsonl" ]; then
    echo "❌ Training data not found at data/hrm_dlp_final/train.jsonl"
    echo "   Make sure you've run the data generation pipeline first."
    exit 1
fi

# Check data sizes
echo "📊 Dataset Statistics:"
echo "   Train: $(wc -l < data/hrm_dlp_final/train.jsonl) examples"
echo "   Val:   $(wc -l < data/hrm_dlp_final/val.jsonl) examples"  
echo "   Test:  $(wc -l < data/hrm_dlp_final/test.jsonl) examples"
echo ""

# Check GPU availability
#if command -v nvidia-smi &> /dev/null; then
#    echo "🔧 GPU Information:"
#    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
#    echo ""
#else
#    echo "⚠️  No GPU detected - training will use CPU (much slower)"
#    echo ""
#fi

# Create checkpoints directory
mkdir -p checkpoints/hrm_dlp

# Run training with configuration
echo "🏃 Launching training..."
python train_dlp.py \
    --config config_dlp_training.json \
    --use_wandb \
    "$@"

echo ""
echo "✅ Training completed!"
echo "📁 Checkpoints saved to: checkpoints/hrm_dlp/"