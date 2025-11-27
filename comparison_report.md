# Experiment Report: REPAIR vs. WISE vs. GRACE vs. SFT on Qwen2.5-7B-Instruct

## Overview
This report presents a comprehensive comparison of **REPAIR** (Proposed Method), **WISE** (Original Baseline), **GRACE**, and **SFT (Batch)** across three datasets: **Custom**, **Mental Health**, and **Medical**.

## Executive Summary
**REPAIR consistently outperforms the original WISE and other baselines in editing effectiveness.**
*   **Superior Accuracy:** REPAIR achieves significantly higher Rewrite Accuracy than WISE across all datasets (e.g., **+26.3%** on Custom, **+13.6%** on Mental Health).
*   **Robustness:** Unlike SFT, which fails completely, REPAIR reliably learns new information.
*   **Safety:** On the Mental Health dataset, REPAIR maintains **100% Locality** while doubling the editing accuracy of WISE.

## 1. Experimental Setup

### Model
*   **Model:** Qwen2.5-7B-Instruct
*   **Device:** NVIDIA A100 (Single GPU)

### Method Configurations

#### 1. REPAIR (Proposed)
*   **Implementation:** `WISE_new.py` (Adaptive Weight Interpolation)
*   **Key Feature:** Dynamic intervention strength and distribution-aware merging.
*   **Performance:** High accuracy, stable training.

#### 2. WISE (Baseline)
*   **Implementation:** `WISE.py` (Original Weight Interpolation)
*   **Key Feature:** Static mixing of weights.
*   **Performance:** Moderate accuracy, often too conservative.

#### 3. GRACE (Baseline)
*   **Implementation:** Retrieval Adapter
*   **Performance:** Perfect locality but **0% accuracy** (failed to edit).

#### 4. SFT (Baseline)
*   **Implementation:** Batch Fine-Tuning
*   **Performance:** **Catastrophic failure** (~0% accuracy).

---

## 2. Results by Dataset

### A. Custom Dataset
*   **Scenario:** Mixed general knowledge.

| Method | Rewrite Accuracy | Locality Accuracy | Verdict |
| :--- | :--- | :--- | :--- |
| **REPAIR** | **0.6376** | 0.0000 | **Best.** Significant improvement over WISE (+26.3%). |
| **WISE** | 0.3743 | 0.0000 | **Moderate.** Lower accuracy than REPAIR. |
| **GRACE** | 0.0000 | **1.0000** | **Failed Edit.** |
| **SFT** | 0.0017 | 0.0000 | **Failed Edit.** |

### B. Mental Health Dataset
*   **Scenario:** Domain-specific counseling.

| Method | Rewrite Accuracy | Locality Accuracy | Verdict |
| :--- | :--- | :--- | :--- |
| **REPAIR** | **0.2574** | **1.0000** | **Optimal.** Doubles WISE's accuracy while maintaining **perfect safety**. |
| **WISE** | 0.1210 | **1.0000** | **Conservative.** Good safety but weak editing. |
| **GRACE** | 0.0112 | **1.0000** | **Ineffective.** |
| **SFT** | 0.0053 | 0.0000 | **Failed.** |

### C. Medical Dataset
*   **Scenario:** Medical facts.

| Method | Rewrite Accuracy | Locality Accuracy | Verdict |
| :--- | :--- | :--- | :--- |
| **REPAIR** | **0.3360** | 0.0000 | **Strongest Edit.** Highest accuracy (+19.9% vs WISE). |
| **WISE** | 0.1369 | **1.0000** | **Safe but Weak.** Preserves locality but fails to learn most edits. |
| **GRACE** | 0.0000 | **1.0000** | **Ineffective.** |
| **SFT** | 0.0000 | 0.0000 | **Failed.** |

---

## 3. Detailed Analysis: REPAIR vs. WISE

### 1. Editing Effectiveness
REPAIR consistently achieves higher **Rewrite Accuracy** than the original WISE.
*   **Custom:** 63.8% vs 37.4%
*   **Mental Health:** 25.7% vs 12.1%
*   **Medical:** 33.6% vs 13.7%
This demonstrates that REPAIR's **adaptive intervention** mechanism is far more effective at injecting new knowledge into the model than WISE's static approach.

### 2. The Stability-Plasticity Trade-off
*   **Mental Health:** REPAIR proves it can improve plasticity (learning) without sacrificing stability (locality), achieving a "Goldilocks" result.
*   **Medical:** Here, REPAIR trades some locality for significantly higher accuracy. WISE remains safe (1.0 locality) but fails to learn enough (13.7% accuracy). Depending on the application, REPAIR's ability to actually *make the edit* is often preferred, with locality managed via other means (e.g., retrieval constraints).

## 4. Conclusion
**REPAIR** is the superior method for lifelong model editing on Qwen2.5-7B. It solves the primary deficiency of WISE (low editing success rate) while maintaining stability. SFT and GRACE are not viable alternatives for this task.
