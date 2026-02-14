#!/bin/bash
# Generate threshold comparison table (JSON, resumable)
# Uses scripts/classify_csv.py for counting
# Output: threshold_comparison.json

OUTPUT_FILE="threshold_comparison.json"

# Get dataset sizes
hist_total=$(($(wc -l < datasets/historical_regex.csv) - 1))
mod_total=$(($(wc -l < datasets/modern_regex.csv) - 1))

# Thresholds to process
thresholds="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# Function to load existing JSON or create empty structure
load_or_initialize() {
    if [ -f "$OUTPUT_FILE" ]; then
        # Check if it's valid JSON with results array
        if python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); assert 'results' in d" 2>/dev/null; then
            echo "Resuming from existing $OUTPUT_FILE"
            python3 -c "import json; print(json.dumps(json.load(open('$OUTPUT_FILE'))['results'], indent=2))"
            return
        fi
    fi
    # Initialize empty
    echo "[]"
}

# Function to get already completed thresholds
get_completed_thresholds() {
    if [ -f "$OUTPUT_FILE" ] && python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); assert 'results' in d" 2>/dev/null; then
        python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); print(' '.join(str(r['threshold']) for r in d['results']))"
    else
        echo ""
    fi
}

# Build list of thresholds to process
completed=$(get_completed_thresholds)
to_process=""
for thresh in $thresholds; do
    # Check if this threshold is in completed list
    if ! echo "$completed" | grep -qw "$thresh"; then
        to_process="$to_process $thresh"
    fi
done

# Remove leading space
to_process=$(echo "$to_process" | xargs)

if [ -z "$to_process" ]; then
    echo "All thresholds already processed!"
else
    echo "Processing thresholds:$to_process"
fi

# Initialize or load existing results
results_json=$(load_or_initialize)

# Process each threshold
for thresh in $thresholds; do
    # Skip if already completed
    if echo "$completed" | grep -qw "$thresh"; then
        echo "Skipping threshold $thresh (already done)"
        continue
    fi

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
        pass_rate="0.0"
    fi

    # Create JSON object for this threshold
    thresh_json=$(python3 -c "
import json
obj = {
    'threshold': $thresh,
    'precision': $precision,
    'recall': $recall,
    'accuracy': $accuracy,
    'f1': $f1,
    'tp': $tp,
    'fn': $fn,
    'fp': $fp,
    'tn': $tn,
    'historical_pass': $hist_result,
    'historical_total': $hist_total,
    'modern_pass': $mod_result,
    'modern_total': $mod_total,
    'pass_rate': $pass_rate
}
print(json.dumps(obj))
")

    # Append to results (read existing, add new, write back)
    if [ -f "$OUTPUT_FILE" ] && python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); assert 'results' in d" 2>/dev/null; then
        # Load existing, append new
        python3 << PYEOF
import json

with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)

results = data['results']

# Check if threshold already exists
exists = False
for r in results:
    if r['threshold'] == $thresh:
        exists = True
        break

if not exists:
    results.append($thresh_json)
    # Sort by threshold
    results.sort(key=lambda x: x['threshold'])

with open('$OUTPUT_FILE', 'w') as f:
    json.dump({'results': results, 'generated': '$(date)'}, f, indent=2)

print("Updated $OUTPUT_FILE")
PYEOF
    else
        # Create new file
        python3 << PYEOF
import json
data = {
    'results': [$thresh_json],
    'generated': '$(date)',
    'datasets': {
        'historical': {'file': 'datasets/historical_regex.csv', 'total': $hist_total},
        'modern': {'file': 'datasets/modern_regex.csv', 'total': $mod_total}
    }
}
with open('$OUTPUT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
print("Created $OUTPUT_FILE")
PYEOF
    fi
done

echo ""
echo "Done! Output saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
