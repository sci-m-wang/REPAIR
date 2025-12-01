#!/bin/bash

# Wait for ELDER to finish
echo "Waiting for ELDER to finish..."
while pgrep -f "run_wise_editing.py" > /dev/null; do
    sleep 60
done
echo "ELDER finished."

# Run Sensitivity Analysis (0.4 - 0.9)
echo "Starting Sensitivity Analysis..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/threshold/run_sensitivity.py > rebuttal_experiments_final/logs/sensitivity_resume.log 2>&1
echo "Sensitivity Analysis finished."

# Run Reasoning Locality (Restart)
echo "Starting Reasoning Locality..."
CUDA_VISIBLE_DEVICES=1 uv run rebuttal_experiments_final/reasoning/run_reasoning_locality.py > rebuttal_experiments_final/logs/reasoning_restart.log 2>&1
echo "Reasoning Locality finished."
