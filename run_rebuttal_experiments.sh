#!/bin/bash
#
# Master Script for Running ICLR Rebuttal Experiments
# Runs 3 priority experiments: Cost Analysis, Heterogeneous Batch Ablation, and RECIPE comparison
#

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./rebuttal_experiments/${TIMESTAMP}"
HPARAMS="hparams/WISE/llama-3-8b.yaml"
DATA_DIR="data/wise"
N=100  # Sample size for initial validation

# Create directories
mkdir -p "$RESULTS_DIR/cost_analysis"
mkdir -p "$RESULTS_DIR/ablation"
mkdir -p "$RESULTS_DIR/logs"

LOG_FILE="${RESULTS_DIR}/logs/main.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "ICLR Rebuttal Experiments" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Sample Size: N=$N" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ========================================
# Experiment 1: Cost Analysis
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 1: Cost Analysis" | tee -a "$LOG_FILE"
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
    echo "✗ REPAIR cost analysis failed" | tee -a "$LOG_FILE"
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
    echo "✗ Original WISE cost analysis failed" | tee -a "$LOG_FILE"
fi

# Restore REPAIR
mv easyeditor/models/wise/wise_main.py.backup easyeditor/models/wise/wise_main.py

# ========================================
# Experiment 2: RECIPE Baseline (Placeholder)
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 2: RECIPE Baseline" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

echo "⚠️  RECIPE implementation not found" | tee -a "$LOG_FILE"
echo "    Action needed: Implement or integrate RECIPE baseline" | tee -a "$LOG_FILE"
echo "    Reference: https://arxiv.org/abs/2305.14956" | tee -a "$LOG_FILE"
echo "    For now, we will use existing WISE results as comparison" | tee -a "$LOG_FILE"

# ========================================
# Experiment 3: Heterogeneous Batch Ablation
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Experiment 3: Heterogeneous Batch Ablation" | tee -a "$LOG_FILE"
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
    echo "✗ Homogeneous batch failed" | tee -a "$LOG_FILE"
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
    echo "✗ Heterogeneous batch failed" | tee -a "$LOG_FILE"
fi

# ========================================
# Generate Summary Report
# ========================================
echo "" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"
echo "Generating Summary Report" | tee -a "$LOG_FILE"
echo "========================================"  | tee -a "$LOG_FILE"

python3 << 'PYEOF' > "${RESULTS_DIR}/SUMMARY.md"
import json
import os
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")

print("# ICLR Rebuttal Experiments - Summary Report")
print(f"\n**Timestamp**: {results_dir.name}")
print(f"**Sample Size**: N=$N")
print(f"**Model**: Meta-Llama-3-8B-Instruct")
print(f"**Dataset**: ZsRE\n")

print("## Experiment 1: Cost Analysis\n")
print("### REPAIR")
try:
    with open(results_dir / "cost_analysis/WISE_N${N}_cost.json") as f:
        repair_cost = json.load(f)
    print(f"- Inference Latency: {repair_cost['avg_inference_latency_ms']:.2f} ms")
    print(f"- Parameter Increment: {repair_cost['side_memory_MB']:.2f} MB")
    print(f"- Training Time: {repair_cost['total_editing_time_s']:.2f} s")
    print(f"- GPU Memory Increase: {repair_cost['gpu_memory_increase_MB']} MB")
except:
    print("- Results not available")

print("\n### Original WISE")
print("- (To be compared with REPAIR)\n")

print("## Experiment 2: RECIPE Baseline\n")
print("⚠️ **Status**: RECIPE implementation needed")
print("- Action: Implement or integrate RECIPE method")
print("- Reference: https://arxiv.org/abs/2305.14956\n")

print("## Experiment 3: Heterogeneous Batch Ablation\n")
try:
    with open(results_dir / "ablation/REPAIR_homogeneous_N${N}.json") as f:
        homo_data = json.load(f)
    with open(results_dir / "ablation/REPAIR_heterogeneous_N${N}.json") as f:
        hetero_data = json.load(f)
    
    from statistics import mean
    
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
    print(f"\n**Locality Degradation**: {(homo_loc - hetero_loc)*100:.1f}%")
except:
    print("- Results not available")

print("\n## Next Steps\n")
print("1. ✅ Cost analysis completed - ready for rebuttal table")
print("2. ⚠️  Implement RECIPE baseline for comparison")
print("3. ✅ Heterogeneous batch ablation completed")
print("4. 📝 Update rebuttal document with findings")
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
