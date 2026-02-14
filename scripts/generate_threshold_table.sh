#!/bin/bash
# Generate threshold comparison table
# Uses scripts/classify_csv.py for counting
# Output: threshold_comparison.md

OUTPUT_FILE="threshold_comparison.md"

echo "# FastText Climate Classifier - Threshold Comparison" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Get dataset sizes
hist_total=$(($(wc -l < datasets/historical_regex.csv) - 1))
mod_total=$(($(wc -l < datasets/modern_regex.csv) - 1))

# Table header
echo "| Threshold | Precision | Recall | Accuracy | F1 | TP | FN | FP | TN | Historical (pass/total) | Modern (pass/total) | Pass Rate |" >> "$OUTPUT_FILE"
echo "|-----------|-----------|--------|----------|----|----|----|----|----|------------------------|-------------------|-----------|" >> "$OUTPUT_FILE"

# Loop through thresholds 0.1 to 0.9
for thresh in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "Processing threshold $thresh..."

    # Get validation metrics using Python
    metrics=$(python3 -c "
import sys
sys.path.insert(0, '.')
from src.fasttext_trainer import evaluate_classifier, apply_predict_patch
import fasttext

apply_predict_patch()
model = fasttext.load_model('models/fasttext_climate.bin')
m = evaluate_classifier(model, 'data/fasttext_valid.txt', threshold=$thresh)

print(f\"{m['climate_precision']:.3f}|{m['climate_recall']:.3f}|{m['accuracy']:.3f}|{m['climate_f1']:.3f}|{m['confusion_matrix']['tp']}|{m['confusion_matrix']['fn']}|{m['confusion_matrix']['fp']}|{m['confusion_matrix']['tn']}\")
" 2>/dev/null)

    # Use classify_csv.py to count climate rows - extract just the number
    hist_result=$(python3 scripts/classify_csv.py -i datasets/historical_regex.csv --mode count --threshold $thresh 2>&1 | grep -E "Climate \(" | grep -oP '\d+(?=\s+\()' | head -1)
    mod_result=$(python3 scripts/classify_csv.py -i datasets/modern_regex.csv --mode count --threshold $thresh 2>&1 | grep -E "Climate \(" | grep -oP '\d+(?=\s+\()' | head -1)

    # Fallback if grep fails
    hist_result=${hist_result:-0}
    mod_result=${mod_result:-0}

    # Parse metrics
    IFS='|' read -r precision recall accuracy f1 tp fn fp tn <<< "$metrics"

    # Calculate pass rate
    total_pass=$((hist_result + mod_result))
    total_rows=$((hist_total + mod_total))
    if [ "$total_rows" -gt 0 ]; then
        pass_rate=$(awk "BEGIN {printf \"%.1f\", ($total_pass/$total_rows)*100}")
    else
        pass_rate="N/A"
    fi

    # Add row to table
    echo "| $thresh | $precision | $recall | $accuracy | $f1 | $tp | $fn | $fp | $tn | $hist_result / $hist_total | $mod_result / $mod_total | $pass_rate% |" >> "$OUTPUT_FILE"
done

echo "" >> "$OUTPUT_FILE"
echo "## Interpretation" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "- **Validation metrics** computed on \`data/fasttext_valid.txt\` (10% held-out from training)" >> "$OUTPUT_FILE"
echo "- **Historical/Modern** = rows passing FastText filter at given threshold / total rows" >> "$OUTPUT_FILE"
echo "- **Pass Rate** = (Historical + Modern passing) / (Historical + Modern total)" >> "$OUTPUT_FILE"
echo "- Historical dataset: \`datasets/historical_regex.csv\` ($hist_total rows)" >> "$OUTPUT_FILE"
echo "- Modern dataset: \`datasets/modern_regex.csv\` ($mod_total rows)" >> "$OUTPUT_FILE"

echo ""
echo "Done! Output saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
