#!/bin/bash

# 1. ELDER Baseline
echo "Starting ELDER Baseline..."
rm rebuttal_experiments_final/logs/elder_*.log
for dataset in ZsRE Hallucination Temporal; do
    CUDA_VISIBLE_DEVICES=1 uv run examples/run_wise_editing.py \
        --editing_method ELDER \
        --hparams_dir hparams/ELDER/qwen2.5-7b.yaml \
        --data_dir data/wise \
        --data_type ${dataset} \
        --ds_size 100 \
        --output_file "rebuttal_experiments_final/elder/ELDER_${dataset}_N100.json" \
        > rebuttal_experiments_final/logs/elder_${dataset}.log 2>&1
done
echo "ELDER Baseline finished."

# 2. Pruning Sensitivity (Resume 0.4 - 0.9)
echo "Starting Sensitivity Analysis..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/threshold/run_sensitivity.py > rebuttal_experiments_final/logs/sensitivity_final.log 2>&1
echo "Sensitivity Analysis finished."

# 3. Reasoning Locality (Restart)
echo "Starting Reasoning Locality..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/reasoning/run_reasoning_locality.py > rebuttal_experiments_final/logs/reasoning_final.log 2>&1
echo "Reasoning Locality finished."

# 4. Similarity Analysis
echo "Starting Similarity Analysis..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/similarity/analyze_batch_similarity.py > rebuttal_experiments_final/logs/similarity.log 2>&1
echo "Similarity Analysis finished."
