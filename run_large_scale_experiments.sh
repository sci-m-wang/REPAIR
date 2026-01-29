#!/bin/bash
#
# Large-Scale Rebuttal Experiments (N=1000) with RECIPE Baseline
#

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./rebuttal_experiments_N1000/${TIMESTAMP}"
HPARAMS="hparams/WISE/llama-3-8b.yaml"
DATA_DIR="data/wise"
N=1000  # Large-scale experiments

# RECIPE configuration
RECIPE_DIR="/tmp/RECIPE"
RECIPE_TRAIN_DIR="$HOME/RECIPE_baseline"

# Create directories
mkdir -p "$RESULTS_DIR/cost_analysis"
mkdir -p "$RESULTS_DIR/ablation"
mkdir -p "$RESULTS_DIR/recipe"
mkdir -p "$RESULTS_DIR/logs"

LOG_FILE="${RESULTS_DIR}/logs/main.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Large-Scale ICLR Rebuttal Experiments" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Sample Size: N=$N" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ========================================
# Experiment 1: Cost Analysis @ N=1000
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 1: Cost Analysis @ N=$N" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

echo "Running REPAIR cost analysis..." | tee -a "$LOG_FILE"
if uv run examples/run_cost_analysis.py \
    --hparams_dir "$HPARAMS" \
    --data_dir "$DATA_DIR" \
    --ds_size $N \
    --output_dir "${RESULTS_DIR}/cost_analysis" \
    > "${RESULTS_DIR}/logs/cost_repair.log" 2>&1; then
    echo "✓ REPAIR cost analysis completed" | tee -a "$LOG_FILE"
else
    echo "✗ REPAIR cost analysis failed - check log" | tee -a "$LOG_FILE"
fi

echo "Running Original WISE cost analysis..." | tee -a "$LOG_FILE"
# Switch to original WISE
cp easyeditor/models/wise/wise_main.py easyeditor/models/wise/wise_main.py.backup
sed -i 's/from \.WISE_new import WISE/from .WISE import WISE/' easyeditor/models/wise/wise_main.py

if uv run examples/run_cost_analysis.py \
    --hparams_dir "$HPARAMS" \
    --data_dir "$DATA_DIR" \
    --ds_size $N \
    --output_dir "${RESULTS_DIR}/cost_analysis" \
    > "${RESULTS_DIR}/logs/cost_wise.log" 2>&1; then
    echo "✓ Original WISE cost analysis completed" | tee -a "$LOG_FILE"
else
    echo "✗ Original WISE cost analysis failed - check log" | tee -a "$LOG_FILE"
fi

# Restore REPAIR
mv easyeditor/models/wise/wise_main.py.backup easyeditor/models/wise/wise_main.py

# ========================================
# Experiment 2: RECIPE Baseline @ N=1000
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 2: RECIPE Baseline @ N=$N" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

if [ -d "$RECIPE_DIR" ]; then
    echo "Setting up RECIPE..." | tee -a "$LOG_FILE"
    
    # Copy RECIPE to working directory if not exists
    if [ ! -d "$RECIPE_TRAIN_DIR" ]; then
        cp -r "$RECIPE_DIR" "$RECIPE_TRAIN_DIR"
        echo "RECIPE copied to $RECIPE_TRAIN_DIR" | tee -a "$LOG_FILE"
    fi
    
    cd "$RECIPE_TRAIN_DIR"
    
    # Install dependencies
    echo "Installing RECIPE dependencies..." | tee -a "$LOG_FILE"
    pip install -r requirement.txt > "${RESULTS_DIR}/logs/recipe_install.log" 2>&1 || true
    
    # Train RECIPE (this may take a while)
    echo "Training RECIPE (this may take several hours)..." | tee -a "$LOG_FILE"
    if python train_recipe.py \
        -mn 'llama-7b' \
        -dn 'zsre' \
        > "${RESULTS_DIR}/logs/recipe_train.log" 2>&1; then
        echo "✓ RECIPE training completed" | tee -a "$LOG_FILE"
        
        # Test RECIPE
        echo "Testing RECIPE @ N=$N..." | tee -a "$LOG_FILE"
        CHECKPOINT=$(ls -t train_records/recipe/llama-7b/*/checkpoints/*.pt 2>/dev/null | head -1)
        
        if [ -n "$CHECKPOINT" ]; then
            if python test_recipe.py \
                -en 'recipe' \
                -mn 'llama-7b' \
                -et 'sequential' \
                -dvc 'cuda:0' \
                -ckpt "$CHECKPOINT" \
                -dn 'zsre' \
                -edn $N \
                > "${RESULTS_DIR}/logs/recipe_test.log" 2>&1; then
                echo "✓ RECIPE testing completed" | tee -a "$LOG_FILE"
                
                # Copy results
                cp -r eval_results/recipe "${RESULTS_DIR}/recipe/" 2>/dev/null || true
            else
                echo "✗ RECIPE testing failed" | tee -a "$LOG_FILE"
            fi
        else
            echo "✗ RECIPE checkpoint not found" | tee -a "$LOG_FILE"
        fi
    else
        echo "✗ RECIPE training failed - check log" | tee -a "$LOG_FILE"
    fi
    
    cd - > /dev/null
else
    echo "⚠️  RECIPE not found at $RECIPE_DIR" | tee -a "$LOG_FILE"
    echo "    Skipping RECIPE experiments" | tee -a "$LOG_FILE"
fi

# ========================================
# Experiment 3: Heterogeneous Batch Ablation @ N=1000
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 3: Heterogeneous Batch Ablation @ N=$N" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

echo "Running HOMOGENEOUS batch (baseline)..." | tee -a "$LOG_FILE"
if uv run examples/run_ablation_heterogeneous.py \
    --hparams_dir "$HPARAMS" \
    --data_dir "$DATA_DIR" \
    --ds_size $N \
    --output_dir "${RESULTS_DIR}/ablation" \
    > "${RESULTS_DIR}/logs/ablation_homogeneous.log" 2>&1; then
    echo "✓ Homogeneous batch completed" | tee -a "$LOG_FILE"
else
    echo "✗ Homogeneous batch failed - check log" | tee -a "$LOG_FILE"
fi

echo "Running HETEROGENEOUS batch (ablation)..." | tee -a "$LOG_FILE"
if uv run examples/run_ablation_heterogeneous.py \
    --hparams_dir "$HPARAMS" \
    --data_dir "$DATA_DIR" \
    --ds_size $N \
    --heterogeneous \
    --output_dir "${RESULTS_DIR}/ablation" \
    > "${RESULTS_DIR}/logs/ablation_heterogeneous.log" 2>&1; then
    echo "✓ Heterogeneous batch completed" | tee -a "$LOG_FILE"
else
    echo "✗ Heterogeneous batch failed - check log" | tee -a "$LOG_FILE"
fi

# ========================================
# Generate Summary Report
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Generating Summary Report" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

python3 << PYEOF > "${RESULTS_DIR}/SUMMARY.md"
import json
from pathlib import Path
from statistics import mean

results_dir = Path("${RESULTS_DIR}")

print("# Large-Scale Rebuttal Experiments - Summary")
print(f"\n**Sample Size**: N=$N")
print(f"**Model**: Meta-Llama-3-8B-Instruct")
print(f"**Dataset**: ZsRE\n")

print("## Experiment 1: Cost Analysis\n")

# REPAIR cost
try:
    with open(results_dir / "cost_analysis/WISE_N${N}_cost.json") as f:
        repair_cost = json.load(f)
    print("### REPAIR")
    print(f"- Inference Latency: {repair_cost.get('avg_inference_latency_ms', 'N/A')} ms")
    print(f"- Side Memory: {repair_cost.get('side_memory_MB', 'N/A')} MB")
    print(f"- Training Time: {repair_cost.get('total_editing_time_s', 'N/A')} s")
    print(f"- GPU Memory Increase: {repair_cost.get('gpu_memory_increase_MB', 'N/A')} MB\n")
except:
    print("### REPAIR: Results not available\n")

print("### Original WISE")
print("- (Compare with REPAIR above)\n")

print("## Experiment 2: RECIPE Baseline\n")
recipe_results = list((results_dir / "recipe").glob("*.json"))
if recipe_results:
    print(f"- Results available: {len(recipe_results)} files")
    print("- See detailed results in recipe/ directory\n")
else:
    print("- Status: Training in progress or not completed\n")

print("## Experiment 3: Heterogeneous Batch Ablation\n")

try:
    with open(results_dir / "ablation/REPAIR_homogeneous_N${N}.json") as f:
        homo_data = json.load(f)
    with open(results_dir / "ablation/REPAIR_heterogeneous_N${N}.json") as f:
        hetero_data = json.load(f)
    
    def calc_metrics(data):
        rewrite = []
        locality = []
        for case in data:
            rewrite.extend(case['post']['rewrite_acc'])
            locality.extend(case['post']['locality']['neighborhood_acc'])
        return mean(rewrite), mean(locality)
    
    homo_rewrite, homo_loc = calc_metrics(homo_data)
    hetero_rewrite, hetero_loc = calc_metrics(hetero_data)
    
    print("| Batch Type | Rewrite Acc | Locality |")
    print("|------------|-------------|----------|")
    print(f"| Homogeneous | {homo_rewrite:.3f} | {homo_loc:.3f} |")
    print(f"| Heterogeneous | {hetero_rewrite:.3f} | {hetero_loc:.3f} |")
    print(f"\n**Locality Degradation**: {(homo_loc - hetero_loc)/homo_loc*100:.1f}%\n")
except:
    print("- Results not available\n")

print("## Files")
print(f"- Results: {results_dir}")
print(f"- Logs: {results_dir}/logs/")
PYEOF

echo "Summary report generated: ${RESULTS_DIR}/SUMMARY.md" | tee -a "$LOG_FILE"

# ========================================
# Final Summary
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ALL EXPERIMENTS COMPLETED" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To view the summary:" | tee -a "$LOG_FILE"
echo "  cat ${RESULTS_DIR}/SUMMARY.md" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Note: RECIPE training may take several hours." | tee -a "$LOG_FILE"
echo "You can monitor progress with:" | tee -a "$LOG_FILE"
echo "  tail -f ${RESULTS_DIR}/logs/recipe_train.log" | tee -a "$LOG_FILE"
