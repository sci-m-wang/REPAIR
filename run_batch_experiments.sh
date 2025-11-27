#!/bin/bash
mkdir -p logs

echo "Starting Batch SFT on Mental Health dataset (Full) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup uv run run_comparison.py --method FT --hparams hparams/FT/qwen2.5-7b_batch.yaml --data_path data/wise/mental_health_full.json --ds_size 100000 --data_type mental_health --batch_sft > logs/mental_health_batch_sft.log 2>&1 &
echo "Mental Health experiment running in background (PID $!). Log: logs/mental_health_batch_sft.log"

echo "Starting Batch SFT on Medical dataset (Full) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup uv run run_comparison.py --method FT --hparams hparams/FT/qwen2.5-7b_batch.yaml --data_path data/wise/medical_full.json --ds_size 100000 --data_type medical --batch_sft > logs/medical_batch_sft.log 2>&1 &
echo "Medical experiment running in background (PID $!). Log: logs/medical_batch_sft.log"
