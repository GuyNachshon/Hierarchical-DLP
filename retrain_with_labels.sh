#!/bin/bash
# HRM-DLP Retraining with Proper Labels
# Fixes the discrimination issue by training with ground truth labels

echo "🚀 HRM-DLP Retraining with Labels"
echo "================================="

# Verify labeled data exists
echo "📊 Checking labeled datasets..."
for split in train val test; do
    file="data/hrm_dlp_final/${split}_labeled.jsonl"
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        echo "   ✅ $split: $count examples"
        
        # Quick label verification
        label_check=$(head -1 "$file" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    labels = data.get('labels', {})
    if len(labels) == 4:
        print('VALID')
    else:
        print('INVALID')
except:
    print('ERROR')
")
        echo "      Labels: $label_check"
    else
        echo "   ❌ Missing: $file"
        echo "   Run: python generate_missing_labels.py"
        exit 1
    fi
done

# Show expected improvements
echo ""
echo "🎯 Expected Training Improvements:"
echo "   Before: Risk scores clustered around 0.51-0.52"
echo "   After:  Risk scores spanning 0.1-0.9 range"
echo "   Before: All decisions were WARN/MEDIUM"
echo "   After:  Proper BLOCK/WARN/ALLOW distribution"
echo "   Before: No discrimination between examples"
echo "   After:  Clear risk level distinctions"
echo ""
echo "✅ Schema Fix Applied:"
echo "   • Fixed Pydantic schema to accept float labels (0.0-1.0)"
echo "   • BCEWithLogitsLoss supports nuanced risk scoring"
echo "   • No more validation errors during training"

# Backup previous checkpoint if exists
if [ -d "checkpoints/hrm_dlp" ]; then
    echo ""
    echo "💾 Backing up previous checkpoint..."
    timestamp=$(date +"%Y%m%d_%H%M%S")
    mv "checkpoints/hrm_dlp" "checkpoints/hrm_dlp_unlabeled_$timestamp"
    echo "   Saved to: checkpoints/hrm_dlp_unlabeled_$timestamp"
fi

echo ""
echo "🏃 Starting training with labeled data..."
echo "   Config: config_dlp_training.json (updated for labeled data)"
echo "   Expected duration: ~10 epochs"
echo "   Monitor for: Decreasing loss AND increasing score discrimination"

# Launch training
python train_dlp.py \
    --config config_dlp_training.json \
    --use_wandb \
    "$@"

echo ""
echo "✅ Training completed!"
echo ""
echo "🧪 Next Steps:"
echo "   1. Test discrimination: python test_model.py --synthetic"
echo "   2. Run diagnostics: python diagnose_model.py"
echo "   3. Validate on examples: python validate_labels.py"
echo "   4. Compare before/after behavior"
echo ""
echo "📊 Expected Results:"
echo "   - Risk scores should now vary meaningfully between examples"
echo "   - High-risk emails should get high scores (>0.7)"
echo "   - Low-risk emails should get low scores (<0.3)"
echo "   - No need for aggressive calibration"