# REPAIR vs Original WISE - Full Comparison Guide

## Quick Start

### 1. Run in Background (Recommended)
```bash
cd /root/REPAIR
nohup ./run_full_comparison.sh > full_comparison.log 2>&1 &
```

### 2. Monitor Progress
```bash
# Watch the main log
tail -f experiment_logs/*/main.log

# Or watch the background log
tail -f full_comparison.log
```

### 3. Check Status
```bash
# See running processes
ps aux | grep run_wise_editing

# Check latest results
ls -lht experiment_results/*/
```

## What This Script Does

The script will automatically:

1. **Test Original WISE** on 6 datasets:
   - ZsRE_Full (18,713 samples) - Main benchmark
   - ZsRE_New (541 samples) - New test set
   - Hallucination_Edit (4,201 samples)
   - Hallucination_Train (2,143 samples)
   - Temporal_Edit (601 samples)
   - Temporal_Train (601 samples)

2. **Test REPAIR (WISE_new)** on the same datasets

3. **Generate comparison report** with all metrics

## Expected Runtime

- **Small datasets** (600 samples): ~30-60 minutes each
- **Medium datasets** (2,000-4,000 samples): ~2-4 hours each
- **Large dataset** (18,713 samples): ~8-12 hours
- **Total estimated time**: ~20-30 hours for all experiments

## Output Structure

```
experiment_results/YYYYMMDD_HHMMSS/
├── SUMMARY.md                           # Overview and instructions
├── OriginalWISE_ZsRE_Full_results.json
├── REPAIR_ZsRE_Full_results.json
├── OriginalWISE_ZsRE_New_results.json
├── REPAIR_ZsRE_New_results.json
└── ... (more result files)

experiment_logs/YYYYMMDD_HHMMSS/
├── main.log                             # Main progress log
├── OriginalWISE_ZsRE_Full.log
├── REPAIR_ZsRE_Full.log
└── ... (individual experiment logs)
```

## Stopping the Experiment

```bash
# Find the process
ps aux | grep run_full_comparison

# Kill it (replace PID with actual process ID)
kill <PID>

# Or kill all related processes
pkill -f run_wise_editing
```

## Resuming After Interruption

The script doesn't have built-in resume capability. If interrupted:

1. Check which experiments completed in `experiment_results/`
2. Edit `run_full_comparison.sh` to remove completed datasets from the `DATASETS` array
3. Re-run the script

## Analyzing Results

After completion, use the Python script in `SUMMARY.md` to generate a comparison table:

```bash
cd experiment_results/<timestamp>/
python analyze_results.py  # (create this based on SUMMARY.md template)
```

## Troubleshooting

### Out of Memory
- Reduce batch size in `hparams/WISE/llama-3-8b.yaml`
- Run datasets sequentially instead of all at once

### Disk Space
- Each result file is ~1-10 MB
- Logs can be large for big datasets
- Ensure at least 10 GB free space

### GPU Issues
- Monitor GPU usage: `nvidia-smi -l 1`
- Check CUDA errors in individual logs

## Alternative: Run Single Dataset

To test just one dataset first:

```bash
# Test on small dataset
./run_full_comparison.sh

# Then edit the script to comment out other datasets
# and run again
```

## Contact

If you encounter issues, check:
1. Individual experiment logs in `experiment_logs/`
2. GPU memory usage
3. Disk space availability
