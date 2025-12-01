#!/bin/bash

# 1. Pruning Sensitivity (Resume 0.4 - 0.9)
echo "Resuming Sensitivity Analysis..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/threshold/run_sensitivity.py > rebuttal_experiments_final/logs/sensitivity_cleanup.log 2>&1
echo "Sensitivity Analysis finished."

# 2. ELDER Baseline (Retry with fixed YAML)
echo "Retrying ELDER Baseline..."
rm rebuttal_experiments_final/logs/elder_retry_*.log
for dataset in ZsRE Hallucination Temporal; do
    CUDA_VISIBLE_DEVICES=1 uv run examples/run_wise_editing.py \
        --editing_method ELDER \
        --hparams_dir hparams/qwen2.5-7b-fixed.yaml \
        --data_dir data/wise \
        --data_type ${dataset} \
        --ds_size 100 \
        --output_file "rebuttal_experiments_final/elder/ELDER_${dataset}_N100_fixed.json" \
        > rebuttal_experiments_final/logs/elder_retry_${dataset}.log 2>&1
done
echo "ELDER Baseline Retry finished."
