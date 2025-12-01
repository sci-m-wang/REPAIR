# Task: Reproduce ELDER

- [ ] Rebuttal Experiments
    - [/] **Experiment 1: ELDER Baseline** <!-- id: 0 -->
    - [x] Implement `run_elder.py` (or adapt `run_wise_editing.py`) <!-- id: 1 -->
    - [x] Fix bugs (`ElderHyperParams`, `key_id`, device, `IN_EDIT_SCOPE`) <!-- id: 2 -->
    - [/] Run on ZsRE, Hallucination, Temporal (Queued in `run_cleanup.sh`) <!-- id: 3 -->
- [/] **Experiment 2: Pruning Threshold Sensitivity** <!-- id: 4 -->
    - [x] Implement `run_sensitivity.py` <!-- id: 5 -->
    - [x] Run thresholds 0.1, 0.2, 0.3 (Completed) <!-- id: 6 -->
    - [/] Run thresholds 0.4 - 0.9 (Running in `run_cleanup.sh`) <!-- id: 7 -->
- [x] **Experiment 3: Reasoning Locality Test** <!-- id: 8 -->
    - [x] Implement `run_reasoning_locality.py` <!-- id: 9 -->
    - [x] Install `lm-eval` <!-- id: 10 -->
    - [x] Fix `tie_weights` bug <!-- id: 11 -->
    - [x] Run Experiment (Completed) <!-- id: 12 -->
- [x] **Experiment 4: Similarity Metric Analysis** <!-- id: 13 -->
    - [x] Implement `analyze_batch_similarity.py` <!-- id: 14 -->
    - [x] Run Analysis (Completed) <!-- id: 15 -->
