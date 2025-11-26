# Large-Scale Experiments (N=1000) - Quick Guide

## 🚀 Running the Experiments

### Start in Background (Recommended)
```bash
cd /root/REPAIR
nohup ./run_large_scale_experiments.sh > large_scale.log 2>&1 &
```

### Monitor Progress
```bash
# Main log
tail -f rebuttal_experiments_N1000/*/logs/main.log

# RECIPE training (may take hours)
tail -f rebuttal_experiments_N1000/*/logs/recipe_train.log

# Background log
tail -f large_scale.log
```

## ⏱️ Expected Runtime

| Experiment | Estimated Time |
|------------|----------------|
| Cost Analysis (REPAIR) | ~2-3 hours |
| Cost Analysis (Original WISE) | ~2-3 hours |
| **RECIPE Training** | **~4-8 hours** |
| RECIPE Testing | ~1 hour |
| Heterogeneous Ablation (x2) | ~4-6 hours |
| **Total** | **~15-25 hours** |

## 📊 What Will Be Generated

### 1. Cost Analysis @ N=1000
- REPAIR: Inference latency, Side Memory size, GPU usage
- Original WISE: Same metrics for comparison

### 2. RECIPE Baseline @ N=1000
- Full training from scratch
- Testing on N=1000 samples
- Direct comparison with REPAIR

### 3. Heterogeneous Batch Ablation @ N=1000
- Homogeneous batches (baseline)
- Heterogeneous batches (ablation)
- Locality degradation analysis

## 📁 Output Structure

```
rebuttal_experiments_N1000/YYYYMMDD_HHMMSS/
├── cost_analysis/
│   ├── WISE_N1000_cost.json
│   └── WISE_N1000_metrics.json
├── recipe/
│   └── (RECIPE results)
├── ablation/
│   ├── REPAIR_homogeneous_N1000.json
│   └── REPAIR_heterogeneous_N1000.json
├── logs/
│   ├── main.log
│   ├── recipe_train.log
│   ├── recipe_test.log
│   └── ...
└── SUMMARY.md
```

## ⚠️ Important Notes

### RECIPE Training
- **First time**: Will train from scratch (~4-8 hours)
- **Checkpoint**: Saved in `/root/RECIPE_baseline/train_records/`
- **Reuse**: Can reuse checkpoint for future experiments

### GPU Memory
- N=1000 requires significant GPU memory
- Monitor with: `watch -n 1 nvidia-smi`
- If OOM, reduce batch size in hparams

### Interruption Recovery
- Each experiment is independent
- Can resume from failed step
- Check logs to see what completed

## 🛑 Stopping Experiments

```bash
# Find process
ps aux | grep run_large_scale

# Kill it
kill <PID>

# Or kill all related
pkill -f run_large_scale
```

## 📊 Quick Results Check

```bash
# Check if experiments completed
ls -lh rebuttal_experiments_N1000/*/

# View summary
cat rebuttal_experiments_N1000/*/SUMMARY.md

# Check specific results
python3 -c "
import json
with open('rebuttal_experiments_N1000/*/ablation/REPAIR_homogeneous_N1000.json') as f:
    data = json.load(f)
    print(f'Loaded {len(data)} cases')
"
```

## 💡 Tips

1. **Run in tmux**: Safer than nohup for long experiments
   ```bash
   tmux new -s experiments
   ./run_large_scale_experiments.sh
   # Ctrl+b, d to detach
   ```

2. **Monitor GPU**: Keep an eye on memory usage
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Parallel RECIPE**: If you have multiple GPUs, can train RECIPE separately
   ```bash
   cd /root/RECIPE_baseline
   CUDA_VISIBLE_DEVICES=1 python train_recipe.py -mn 'llama-7b' -dn 'zsre'
   ```

## 🎯 After Completion

1. Analyze results with provided scripts
2. Generate rebuttal tables
3. Compare with N=100 results for consistency
