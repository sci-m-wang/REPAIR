#!/bin/bash
#
# REPAIR vs Original WISE - Full Dataset Comparison Script (Fixed)
# This script runs comprehensive experiments on all available datasets
#

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./experiment_logs/${TIMESTAMP}"
RESULTS_DIR="./experiment_results/${TIMESTAMP}"
HPARAMS="hparams/WISE/llama-3-8b.yaml"
DATA_DIR="data/wise"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Log file
MAIN_LOG="${LOG_DIR}/main.log"

echo "========================================" | tee -a "$MAIN_LOG"
echo "REPAIR vs Original WISE Comparison" | tee -a "$MAIN_LOG"
echo "Started at: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# Dataset configurations
# Format: "experiment_name:data_type:description"
EXPERIMENTS=(
    "ZsRE:ZsRE:Zero-shot Relation Extraction"
)

# Sample sizes to test (you can add more: 100, 500, 1000, etc.)
SAMPLE_SIZES=(10 100 500)

# Function to run experiment
run_experiment() {
    local method=$1
    local exp_name=$2
    local data_type=$3
    local sample_size=$4
    local description=$5
    
    echo "" | tee -a "$MAIN_LOG"
    echo "----------------------------------------" | tee -a "$MAIN_LOG"
    echo "Running: $method on $exp_name (N=$sample_size)" | tee -a "$MAIN_LOG"
    echo "Description: $description" | tee -a "$MAIN_LOG"
    echo "Started: $(date)" | tee -a "$MAIN_LOG"
    
    # Log file for this experiment
    EXP_LOG="${LOG_DIR}/${method}_${exp_name}_N${sample_size}.log"
    
    # Run the experiment
    if uv run examples/run_wise_editing.py \
        --hparams_dir "$HPARAMS" \
        --data_dir "$DATA_DIR" \
        --data_type "$data_type" \
        --ds_size "$sample_size" \
        --editing_method WISE \
        > "$EXP_LOG" 2>&1; then
        
        echo "✓ Completed successfully" | tee -a "$MAIN_LOG"
        
        # Copy results
        LATEST_OUTPUT=$(ls -t outputs/Meta-Llama-3-8B-Instruct_WISE_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_OUTPUT" ]; then
            cp "$LATEST_OUTPUT" "${RESULTS_DIR}/${method}_${exp_name}_N${sample_size}_results.json"
            echo "  Results saved to: ${RESULTS_DIR}/${method}_${exp_name}_N${sample_size}_results.json" | tee -a "$MAIN_LOG"
            
            # Extract and display key metrics
            if command -v jq &> /dev/null; then
                echo "  Metrics:" | tee -a "$MAIN_LOG"
                jq -r '.post | "    Rewrite Acc: \(.rewrite_acc // "N/A"), Rephrase Acc: \(.rephrase_acc // "N/A"), Locality: \(.locality.neighborhood_acc // "N/A")"' "$LATEST_OUTPUT" 2>/dev/null | tee -a "$MAIN_LOG" || echo "    (Could not parse metrics)" | tee -a "$MAIN_LOG"
            fi
        fi
    else
        echo "✗ Failed - check log: $EXP_LOG" | tee -a "$MAIN_LOG"
    fi
    
    echo "Finished: $(date)" | tee -a "$MAIN_LOG"
}

# Backup current configuration
echo "Backing up configuration..." | tee -a "$MAIN_LOG"
cp easyeditor/models/wise/wise_main.py easyeditor/models/wise/wise_main.py.backup

# ========================================
# Part 1: Run Original WISE on all datasets
# ========================================
echo "" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"
echo "PART 1: Testing Original WISE" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"

# Switch to original WISE
sed -i 's/from \.WISE_new import WISE/from .WISE import WISE/' easyeditor/models/wise/wise_main.py
echo "Switched to Original WISE" | tee -a "$MAIN_LOG"

for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_type description <<< "$exp_config"
    for size in "${SAMPLE_SIZES[@]}"; do
        run_experiment "OriginalWISE" "$exp_name" "$data_type" "$size" "$description"
    done
done

# ========================================
# Part 2: Run REPAIR (WISE_new) on all datasets
# ========================================
echo "" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"
echo "PART 2: Testing REPAIR (WISE_new)" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"

# Switch to REPAIR
sed -i 's/from \.WISE import WISE/from .WISE_new import WISE/' easyeditor/models/wise/wise_main.py
echo "Switched to REPAIR (WISE_new)" | tee -a "$MAIN_LOG"

for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_type description <<< "$exp_config"
    for size in "${SAMPLE_SIZES[@]}"; do
        run_experiment "REPAIR" "$exp_name" "$data_type" "$size" "$description"
    done
done

# ========================================
# Restore configuration
# ========================================
echo "" | tee -a "$MAIN_LOG"
echo "Restoring original configuration..." | tee -a "$MAIN_LOG"
mv easyeditor/models/wise/wise_main.py.backup easyeditor/models/wise/wise_main.py

# ========================================
# Generate summary report
# ========================================
echo "" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"
echo "Generating Summary Report" | tee -a "$MAIN_LOG"
echo "========================================"  | tee -a "$MAIN_LOG"

SUMMARY_FILE="${RESULTS_DIR}/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# REPAIR vs Original WISE - Comparison Results

## Experiment Information
- **Timestamp**: ${TIMESTAMP}
- **Model**: Meta-Llama-3-8B-Instruct
- **Config**: ${HPARAMS}
- **Data Directory**: ${DATA_DIR}

## Experiments Run

| Experiment | Sample Sizes | Description |
|------------|--------------|-------------|
EOF

for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_type description <<< "$exp_config"
    sizes_str=$(IFS=', '; echo "${SAMPLE_SIZES[*]}")
    echo "| $exp_name | $sizes_str | $description |" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << 'EOF'

## Results Files

Results are organized by method and sample size:
- `OriginalWISE_<experiment>_N<size>_results.json`
- `REPAIR_<experiment>_N<size>_results.json`

## Quick Analysis

To compare results, use this Python snippet:

```python
import json
import glob

results = {}
for file in glob.glob('*_results.json'):
    with open(file) as f:
        data = json.load(f)
        parts = file.replace('_results.json', '').split('_')
        method = parts[0]
        exp = '_'.join(parts[1:-1])
        size = parts[-1]
        
        key = f"{method}_{exp}_{size}"
        results[key] = {
            'rewrite_acc': data.get('post', {}).get('rewrite_acc'),
            'rephrase_acc': data.get('post', {}).get('rephrase_acc'),
            'locality': data.get('post', {}).get('locality', {}).get('neighborhood_acc')
        }

import pandas as pd
df = pd.DataFrame(results).T
print(df.to_markdown())
```

## Key Metrics to Compare

1. **Rewrite Accuracy**: How well the edit was applied
2. **Rephrase Accuracy**: Generalization to rephrased queries
3. **Locality**: Preservation of unrelated knowledge (higher is better)

## Expected Findings

Based on previous tests:
- **REPAIR** should show significantly better locality (~90% vs ~0.3%)
- **Original WISE** may have slightly higher rewrite accuracy (~65% vs ~62%)

EOF

echo "Summary report generated: $SUMMARY_FILE" | tee -a "$MAIN_LOG"

# ========================================
# Final summary
# ========================================
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "ALL EXPERIMENTS COMPLETED" | tee -a "$MAIN_LOG"
echo "Finished at: $(date)" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Results directory: $RESULTS_DIR" | tee -a "$MAIN_LOG"
echo "Logs directory: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "To view the summary:" | tee -a "$MAIN_LOG"
echo "  cat $SUMMARY_FILE" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
