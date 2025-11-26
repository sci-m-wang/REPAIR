#!/bin/bash
#
# Quick RECIPE Experiment @ N=100
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$(pwd)/recipe_experiment_N100/${TIMESTAMP}"
N=100

# RECIPE configuration
RECIPE_DIR="/tmp/RECIPE"
RECIPE_TRAIN_DIR="/root/RECIPE_baseline"

# Create directories first
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/results"

LOG_FILE="${RESULTS_DIR}/logs/main.log"

echo "========================================" | tee "$LOG_FILE"
echo "RECIPE Baseline Experiment @ N=$N" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

if [ -d "$RECIPE_DIR" ]; then
    echo "Setting up RECIPE..." | tee -a "$LOG_FILE"
    
    # Copy RECIPE if not exists
    if [ ! -d "$RECIPE_TRAIN_DIR" ]; then
        echo "Copying RECIPE to $RECIPE_TRAIN_DIR..." | tee -a "$LOG_FILE"
        cp -r "$RECIPE_DIR" "$RECIPE_TRAIN_DIR"
        echo "✓ RECIPE copied" | tee -a "$LOG_FILE"
    else
        echo "RECIPE already exists at $RECIPE_TRAIN_DIR" | tee -a "$LOG_FILE"
    fi
    
    cd "$RECIPE_TRAIN_DIR"
    echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
    
    # Install dependencies
    echo "Installing RECIPE dependencies..." | tee -a "$LOG_FILE"
    pip install -r requirement.txt > "${RESULTS_DIR}/logs/install.log" 2>&1 || true
    echo "✓ Dependencies installed" | tee -a "$LOG_FILE"
    
    # Train RECIPE
    echo "Training RECIPE with Llama-3-8B-Instruct @ N=$N (this may take 1-2 hours)..." | tee -a "$LOG_FILE"
    if python train_recipe.py \
        -mn 'llama-3-8b-instruct' \
        -dn 'zsre' \
        > "${RESULTS_DIR}/logs/train.log" 2>&1; then
        echo "✓ RECIPE training completed" | tee -a "$LOG_FILE"
        
        # Test RECIPE
        echo "Testing RECIPE @ N=$N..." | tee -a "$LOG_FILE"
        CHECKPOINT=$(ls -t train_records/recipe/llama-3-8b-instruct/*/checkpoints/*.pt 2>/dev/null | head -1)
        
        if [ -n "$CHECKPOINT" ]; then
            echo "Using checkpoint: $CHECKPOINT" | tee -a "$LOG_FILE"
            
            if python test_recipe.py \
                -en 'recipe' \
                -mn 'llama-3-8b-instruct' \
                -et 'sequential' \
                -dvc 'cuda:0' \
                -ckpt "$CHECKPOINT" \
                -dn 'zsre' \
                -edn $N \
                > "${RESULTS_DIR}/logs/test.log" 2>&1; then
                echo "✓ RECIPE testing completed" | tee -a "$LOG_FILE"
                
                # Copy results
                mkdir -p "${RESULTS_DIR}/results"
                cp -r eval_results/recipe/* "${RESULTS_DIR}/results/" 2>/dev/null || true
                
                echo "" | tee -a "$LOG_FILE"
                echo "Results saved to: ${RESULTS_DIR}/results/" | tee -a "$LOG_FILE"
            else
                echo "✗ RECIPE testing failed - check ${RESULTS_DIR}/logs/test.log" | tee -a "$LOG_FILE"
            fi
        else
            echo "✗ RECIPE checkpoint not found" | tee -a "$LOG_FILE"
        fi
    else
        echo "✗ RECIPE training failed - check ${RESULTS_DIR}/logs/train.log" | tee -a "$LOG_FILE"
    fi
    
    cd - > /dev/null
else
    echo "✗ RECIPE not found at $RECIPE_DIR" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "RECIPE Experiment Completed" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Results: $RESULTS_DIR" | tee -a "$LOG_FILE"
