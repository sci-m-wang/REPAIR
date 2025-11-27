#!/bin/bash
mkdir -p logs

# GRACE Experiments (GPU 1)
echo "Retrying GRACE experiments on GPU 1..."
(
    echo "Running GRACE on Custom..."
    CUDA_VISIBLE_DEVICES=1 uv run run_comparison.py --method GRACE --hparams hparams/GRACE/qwen2.5-7b.yaml --data_path data/wise/custom_qwen_test.json --ds_size 10 --data_type custom > logs/grace_custom.log 2>&1
    
    echo "Running GRACE on Mental Health..."
    CUDA_VISIBLE_DEVICES=1 uv run run_comparison.py --method GRACE --hparams hparams/GRACE/qwen2.5-7b.yaml --data_path data/wise/mental_health_test.json --ds_size 10 --data_type mental_health > logs/grace_mental_health.log 2>&1
    
    echo "Running GRACE on Medical..."
    CUDA_VISIBLE_DEVICES=1 uv run run_comparison.py --method GRACE --hparams hparams/GRACE/qwen2.5-7b.yaml --data_path data/wise/medical_test.json --ds_size 10 --data_type medical > logs/grace_medical.log 2>&1
    
    echo "GRACE experiments completed."
) &

wait
echo "Retry completed."
