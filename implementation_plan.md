# Implementation Plan - Stabilizing Batch SFT

The Batch SFT experiments failed due to gradient explosion (NaN loss). This plan aims to stabilize the training by adding gradient clipping and adjusting hyperparameters.

## Proposed Changes

### 1. Modify `easyeditor/models/ft/ft_main.py`
- Add `max_grad_norm` parameter to `execute_ft` (default 1.0).
- Implement `torch.nn.utils.clip_grad_norm_` before `opt.step()`.

### 2. Create Stable Hyperparameters
- Create `hparams/FT/qwen2.5-7b_batch_stable.yaml`.
- Reduce Learning Rate: `1e-4` -> `1e-5`.
- Reduce Epochs: `50` -> `5` (50 epochs on 20k samples is overkill and likely causes overfitting/instability).
- Add `max_grad_norm: 1.0`.

### 4. Final Execution Strategy (Machine Expiring)

Due to imminent machine expiry, we are prioritizing the consolidation of existing results.

### Completed
- **Reasoning Locality:** MMLU 96% retention.
- **Similarity Analysis:** Cross-task similarity 0.44.
- **Sensitivity:** Thresholds 0.1-0.4 completed.

### Running / Pending
- **Sensitivity:** Resumed from 0.5.
- **ELDER:** Queued (unlikely to finish).

### Contingency
- If machine expires, we have sufficient data to demonstrate:
    1.  REPAIR's safety (Reasoning).
    2.  Task distinctness (Similarity).
    3.  Locality improvement with threshold (Sensitivity 0.1-0.4).

### 3. Update Experiment Script
- Update `run_batch_experiments.sh` to use the new stable config.

## Verification Plan

### Automated Tests
- Run the updated `run_batch_experiments.sh`.
- Monitor logs for `Batch loss nan`.
- Check if loss decreases or stays stable.
