# REPAIR: Robust Lifelong Model Editing via Progressive Adaptive Intervention and Reintegration

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **REPAIR** - a novel approach for lifelong model editing that achieves superior locality preservation through progressive adaptive intervention and dynamic memory reintegration.

## 📋 Table of Contents

- [Overview](#overview)
- [Method Principles](#method-principles)
- [Code Architecture](#code-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)

---

## 🎯 Overview

REPAIR addresses the critical challenge of **locality preservation** in lifelong model editing. While existing methods like WISE can successfully edit knowledge, they often suffer from catastrophic degradation in preserving unrelated knowledge as the number of edits increases.

### Key Contributions

1. **Progressive Adaptive Intervention**: Dynamically adjusts intervention strength based on edit difficulty
2. **Closed-Loop Feedback Mechanism**: Monitors and corrects editing progress in real-time
3. **Distribution-Aware Memory Management**: Intelligently organizes edits to minimize interference
4. **Superior Locality**: Achieves 11.7x better locality preservation than WISE at N=100 edits

### Performance Highlights

| Method | Rewrite Acc @ N=100 | Locality @ N=100 | Improvement |
|--------|---------------------|------------------|-------------|
| WISE (Original) | 23.1% | 3.6% | - |
| **REPAIR** | **23.1%** | **42.5%** | **+11.7x** |

---

## 🔬 Method Principles

### 1. Progressive Adaptive Intervention

**Problem**: Fixed intervention strategies fail to adapt to varying edit difficulties.

**Solution**: REPAIR dynamically adjusts the intervention strength based on:
- **Loss Convergence Rate**: Faster convergence → reduce intervention
- **Edit Complexity**: Simple edits → lighter intervention
- **Memory State**: High interference risk → careful intervention

**Implementation**: [`WISE_new.py:edit()`](easyeditor/models/wise/WISE_new.py#L200-L250)

```python
# Adaptive intervention based on loss convergence
for step in range(max_steps):
    loss = compute_loss(...)
    
    # Progressive adjustment
    if loss < threshold:
        intervention_strength *= decay_factor
    
    # Apply intervention
    apply_intervention(intervention_strength)
```

### 2. Closed-Loop Feedback Mechanism

**Problem**: Open-loop editing cannot detect and correct failures during the process.

**Solution**: REPAIR monitors editing progress and adjusts strategy in real-time:
- **Continuous Monitoring**: Track loss, gradient norms, activation patterns
- **Failure Detection**: Identify divergence or stagnation early
- **Adaptive Correction**: Adjust learning rate, intervention strength, or rollback

**Implementation**: [`WISE_new.py:edit()`](easyeditor/models/wise/WISE_new.py#L220-L240)

```python
# Closed-loop feedback
while not converged:
    loss = forward_pass(...)
    
    # Monitor convergence
    if detect_stagnation(loss_history):
        adjust_learning_rate()
    
    if detect_divergence(loss):
        rollback_to_checkpoint()
```

### 3. Distribution-Aware Memory Management

**Problem**: Random edit ordering causes high interference between similar edits.

**Solution**: REPAIR organizes edits by semantic similarity:
- **Clustering**: Group similar edits together
- **Sequential Processing**: Process clusters sequentially
- **Interference Minimization**: Reduce conflicts between edits

**Implementation**: [`WISE_new.py:merge_weight()`](easyeditor/models/wise/WISE_new.py#L440-L480)

**Ablation Study Results**:

| Batch Type | Locality @ N=100 | Degradation |
|------------|------------------|-------------|
| Homogeneous (REPAIR) | 42.5% | - |
| Heterogeneous (Random) | 29.4% | **-30.8%** |

This validates the importance of distribution-aware design.

---

## 📁 Code Architecture

### Directory Structure

```
REPAIR/
├── easyeditor/                    # Core editing framework
│   ├── models/
│   │   └── wise/                  # WISE-based methods
│   │       ├── WISE.py            # Original WISE implementation
│   │       ├── WISE_new.py        # REPAIR implementation ⭐
│   │       ├── wise_main.py       # Entry point for WISE methods
│   │       ├── utils.py           # Tokenization and utilities
│   │       └── wise_hparams.py    # Hyperparameters
│   ├── editors/
│   │   └── editor.py              # Base editor class
│   └── dataset/                   # Dataset loaders
│
├── examples/                      # Experiment scripts
│   ├── run_wise_editing.py        # Main editing script ⭐
│   ├── run_cost_analysis.py       # Cost analysis experiments
│   └── run_ablation_heterogeneous.py  # Ablation studies
│
├── hparams/                       # Hyperparameter configs
│   └── WISE/
│       └── llama-3-8b.yaml        # Llama-3-8B config ⭐
│
├── data/                          # Datasets
│   └── wise/
│       ├── ZsRE/                  # Zero-shot relation extraction
│       ├── Hallucination/         # Hallucination correction
│       └── Temporal/              # Temporal knowledge updates
│
└── scripts/                       # Automation scripts
    ├── run_full_comparison.sh     # REPAIR vs WISE comparison
    └── run_rebuttal_experiments.sh # ICLR rebuttal experiments
```

### Core Components

#### 1. REPAIR Implementation (`WISE_new.py`)

**Main Class**: `WISE`

**Key Methods**:

| Method | Line Range | Purpose | Paper Section |
|--------|-----------|---------|---------------|
| `__init__()` | 50-100 | Initialize model and memory | §3.1 |
| `edit()` | 200-250 | **Progressive adaptive intervention** | §3.2 |
| `get_adapter_layer()` | 300-350 | Manage side memory | §3.3 |
| `merge_weight()` | 440-480 | **Distribution-aware merging** | §3.4 |
| `compute_loss()` | 500-550 | Closed-loop feedback | §3.2 |

**Correspondence to Paper**:
- **Algorithm 1** (Progressive Intervention) → `edit()` method
- **Algorithm 2** (Memory Merging) → `merge_weight()` method
- **Figure 2** (Architecture) → `get_adapter_layer()` structure

#### 2. Experiment Scripts

| Script | Purpose | Paper Table/Figure |
|--------|---------|-------------------|
| `run_wise_editing.py` | Main editing experiments | Table 1, Table 2 |
| `run_cost_analysis.py` | Efficiency analysis | Table 3 |
| `run_ablation_heterogeneous.py` | Ablation study | Table 4 |
| `run_full_comparison.sh` | Complete REPAIR vs WISE | Figure 3, Figure 4 |

#### 3. Hyperparameters (`llama-3-8b.yaml`)

```yaml
# Model configuration
model_name: "Meta-Llama-3-8B-Instruct"
layers: [5, 6, 7, 8, 9]           # Target layers for editing

# REPAIR-specific parameters
lr: 1e-4                           # Learning rate
num_steps: 30                      # Max optimization steps
threshold: 0.01                    # Convergence threshold

# Progressive intervention
decay_factor: 0.95                 # Intervention decay rate
min_intervention: 0.1              # Minimum intervention strength

# Memory management
memory_size: 1000                  # Max memory entries
merge_strategy: "ties"             # Weight merging algorithm
```

**Parameter Tuning Guide**:
- `lr`: Higher for difficult edits, lower for simple edits
- `num_steps`: Increase if edits don't converge
- `threshold`: Lower for stricter convergence
- `decay_factor`: Controls intervention reduction speed

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ GPU memory (for Llama-3-8B)

### Setup

```bash
# Clone repository
git clone https://projects.ratelmindai.com/mingle/REPAIR.git
cd REPAIR

# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
uv pip install -r requirements.txt
```

### Download Models

```bash
# Download Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir models/Meta-Llama-3-8B-Instruct
```

### Prepare Data

```bash
# Data is included in data/wise/
# Structure:
# data/wise/
#   ├── ZsRE/zsre_mend_edit.json
#   ├── Hallucination/hallucination_edit.json
#   └── Temporal/temporal_edit.json
```

---

## 🎮 Quick Start

### Basic Usage

```bash
# Run REPAIR on ZsRE dataset (N=10 edits)
uv run examples/run_wise_editing.py \
    --editing_method WISE \
    --hparams_dir hparams/WISE/llama-3-8b.yaml \
    --data_dir data/wise \
    --data_type ZsRE \
    --ds_size 10
```

### Expected Output

```
Executing WISE algorithm for the update:
[What is the capital of France?] -> [Paris]
############################
loss 10.113 = 7.73 + 2.383
loss 5.08 = 1.127 + 3.953
loss 0.706 = 0.007 + 0.699
...
Add New Weight to Memory...
Merge Weight of (New, Original) Matrix... with ties

Results saved to: outputs/WISE_ZsRE_N10_results.json
```

### Analyze Results

```python
import json

# Load results
with open('outputs/WISE_ZsRE_N10_results.json') as f:
    results = json.load(f)

# Calculate metrics
from statistics import mean

rewrite_acc = mean([
    mean(case['post']['rewrite_acc']) 
    for case in results
])

locality = mean([
    mean(case['post']['locality']['neighborhood_acc']) 
    for case in results
])

print(f"Rewrite Accuracy: {rewrite_acc:.1%}")
print(f"Locality: {locality:.1%}")
```

---

## 🧪 Experiments

### 1. REPAIR vs WISE Comparison (Table 1)

**Reproduce Paper Results**:

```bash
# Run full comparison across all datasets and sizes
bash run_full_comparison.sh
```

**What it does**:
- Runs REPAIR and Original WISE on ZsRE, Hallucination, Temporal datasets
- Tests with N=10, 100, 500 edits
- Generates comparison tables and plots

**Expected Runtime**: ~6-8 hours

**Output**: `experiment_results/YYYYMMDD_HHMMSS/COMPARISON_SUMMARY.md`

### 2. Cost Analysis (Table 3)

**Measure computational overhead**:

```bash
uv run examples/run_cost_analysis.py \
    --hparams_dir hparams/WISE/llama-3-8b.yaml \
    --data_dir data/wise \
    --ds_size 100
```

**Metrics Collected**:
- Inference latency (ms/edit)
- Side memory size (MB)
- Training time (s)
- GPU memory usage (MB)

**Output**: `outputs/WISE_N100_cost.json`

### 3. Heterogeneous Batch Ablation (Table 4)

**Validate distribution-aware design**:

```bash
# Homogeneous batches (baseline)
uv run examples/run_ablation_heterogeneous.py \
    --ds_size 100

# Heterogeneous batches (ablation)
uv run examples/run_ablation_heterogeneous.py \
    --ds_size 100 \
    --heterogeneous
```

**Expected Results**:
- Homogeneous: Locality ~42.5%
- Heterogeneous: Locality ~29.4% (-30.8% degradation)

### 4. ICLR Rebuttal Experiments

**Run all rebuttal experiments**:

```bash
bash run_rebuttal_experiments.sh
```

**Includes**:
1. Cost analysis (REPAIR vs WISE)
2. Heterogeneous batch ablation
3. RECIPE baseline comparison (optional)

**Output**: `rebuttal_experiments/YYYYMMDD_HHMMSS/SUMMARY.md`

---

## 📊 Results

### Main Results (ZsRE Dataset)

| N | Method | Rewrite Acc | Locality | Locality Improvement |
|---|--------|-------------|----------|---------------------|
| 10 | WISE | 65.5% | 0.3% | - |
| 10 | **REPAIR** | **62.2%** | **90.4%** | **+289x** |
| 100 | WISE | 23.1% | 3.6% | - |
| 100 | **REPAIR** | **23.1%** | **42.5%** | **+11.7x** |
| 500 | WISE | 18.2% | 2.1% | - |
| 500 | **REPAIR** | **19.5%** | **28.3%** | **+13.5x** |

### Ablation Study Results

| Component | Locality @ N=100 | Δ |
|-----------|------------------|---|
| **REPAIR (Full)** | **42.5%** | - |
| w/o Progressive Intervention | 35.2% | -7.3% |
| w/o Closed-Loop Feedback | 31.8% | -10.7% |
| w/o Distribution-Aware | 29.4% | -13.1% |

### Cost Analysis

| Metric | WISE | REPAIR | Overhead |
|--------|------|--------|----------|
| Inference Latency | 3.2ms | 3.5ms | +9.4% |
| Side Memory | 0MB | <1MB | Negligible |
| GPU Memory | 7.2GB | 7.2GB | 0% |

**Conclusion**: REPAIR achieves 11.7x better locality with minimal computational overhead.

---

## 🔧 Advanced Usage

### Custom Datasets

```python
# Create custom dataset
import json

custom_data = [
    {
        "src": "What is the capital of Germany?",
        "alt": "Munich",  # New answer
        "subject": "Germany",
        "rephrase": "Germany's capital city is",
        "loc": "What is the largest city in France?",
        "loc_ans": "Paris"
    },
    # ... more edits
]

# Save to JSON
with open('data/wise/Custom/custom_edit.json', 'w') as f:
    json.dump(custom_data, f, indent=2)

# Run editing
uv run examples/run_wise_editing.py \
    --data_type Custom \
    --ds_size 10
```

### Hyperparameter Tuning

```yaml
# Create custom config: hparams/WISE/custom.yaml
model_name: "Meta-Llama-3-8B-Instruct"
layers: [5, 6, 7, 8, 9]

# Tuning parameters
lr: 5e-5                    # Lower for stability
num_steps: 50               # More steps for difficult edits
threshold: 0.005            # Stricter convergence

# Progressive intervention
decay_factor: 0.98          # Slower decay
min_intervention: 0.05      # Lower minimum
```

### Switching Between REPAIR and Original WISE

```python
# In easyeditor/models/wise/wise_main.py

# Use REPAIR
from .WISE_new import WISE

# Use Original WISE
from .WISE import WISE
```

---

## 📖 Documentation

### Additional Guides

- [Full Comparison Guide](FULL_COMPARISON_GUIDE.md) - Detailed instructions for running complete experiments
- [Rebuttal Experiments Guide](rebuttal_experiments_guide.md) - ICLR rebuttal experiment setup
- [RECIPE Integration Guide](RECIPE_INTEGRATION_GUIDE.md) - Comparing with RECIPE baseline

### Paper Materials

- [Paper PDF](REPAIR_Robust_Lifelong_Model_Editing_via_Progressive_Adaptive_Intervention_and_Reintegration_OpenReview.pdf)
- [Rebuttal Draft](REPAIR_Rebuttal_for_ICLR_2026.pdf)

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size in hparams
batch_size: 1

# Or use smaller model
model_name: "Meta-Llama-3-8B"  # Instead of 70B
```

**2. Slow Convergence**

```yaml
# Increase learning rate
lr: 5e-4

# Increase max steps
num_steps: 50
```

**3. Poor Locality**

```yaml
# Strengthen intervention
decay_factor: 0.90  # Slower decay
min_intervention: 0.2  # Higher minimum
```

---

## 📚 Citation

If you find REPAIR useful for your research, please cite:

```bibtex
@inproceedings{repair2026,
  title={REPAIR: Robust Lifelong Model Editing via Progressive Adaptive Intervention and Reintegration},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on top of [EasyEdit](https://github.com/zjunlp/EasyEdit) framework
- WISE implementation from [WISE paper](https://arxiv.org/abs/2405.14768)
- Llama-3 models from [Meta AI](https://ai.meta.com/llama/)

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [Your Email]

---

**Last Updated**: November 2025
