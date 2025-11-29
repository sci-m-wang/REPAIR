# ICLR审稿人完整需求分析（基于原文）

## 📊 四位审稿人概览

| 审稿人 | 评分 | 态度 | 主要关注 | 实验需求数量 |
|--------|------|------|---------|-------------|
| **G6uc** | **4分** ⚠️ | **边缘拒稿** | 评估不足、计算成本、超参数敏感性 | 7个 |
| **7U7d** | **4分** ⚠️ | **边缘拒稿** | 排版错误、异构批次、RECIPE对比 | 4个 |
| **4dsu** | **6分** ✅ | 边缘接受 | 计算成本、超参数敏感性、假阳性剪枝 | 3个 |
| **oiCK** | **6分** ✅ | 边缘接受 | 拼写错误、成本分析、消融研究 | 4个 |

**关键洞察**: 
- ⚠️ **G6uc和7U7d给了4分（边缘拒稿）** - 他们的要求是**最高优先级**
- ✅ 4dsu和oiCK给了6分（边缘接受） - 他们的要求相对次要

---

## 🔴 Reviewer G6uc - 详细分析

### Weaknesses分类

#### W1: 计算成本问题（观察性，非实验需求）
> "REPAIR add unneglectable amount of additional compute and cost compared to WISE"

**性质**: 这是观察，不是实验需求
**已有**: Rebuttal论文中有throughput分析
**需要**: 可能需要更详细的cost breakdown

#### W2: 中等规模不稳定性（观察性）
> "Transient instability mid-scale: At N≈120"

**性质**: 观察，不需要新实验
**回应**: 在rebuttal中解释这是正常的适应过程

#### W3: 超参数敏感性（观察性）
> "Hyperparameter sensitivity: Thresholds for error filtering..."

**性质**: 观察，但暗示需要敏感性分析
**实验需求**: ✅ 需要敏感性分析实验

#### W4: 评估不足 - 缺少Baseline ⭐⭐⭐
> "Missing some more recent lifelong editing baselines such as sLKE [1], LeMOE [2], and ELDER [3]"

**实验需求**:
- ✅ ELDER - 已实现，可运行
- ❌ sLKE - 未实现
- ❌ LeMOE - 未实现

#### W5: FT设置不当 ⭐⭐
> "The finetuning baseline should adopt the fair setups as discussed in [4,5]. The FT-L, FT-M are ill-defined"

**实验需求**: 
- 需要使用标准LoRA或完整FT设置
- 参考文献[4,5]的建议

#### W6: 缺少更广泛任务的Locality测试 ⭐⭐
> "testing locality regarding reasoning / tool-use ability after sequential editing is meaningful"

**实验需求**:
- Reasoning locality (GSM8K, MMLU)
- Tool-use locality

#### W7: 时间步扩展 ⭐
> "The scaling of timestep is only to 1k. More timesteps can be shown, e.g., up to 5k"

**实验需求**: N=5000实验

#### W8: 写作质量（非实验需求）
> "Writing quality can be improved. For example, Table 2..."

**性质**: 写作改进，不需要实验

### Questions

#### Q1: 改进评估部分
> "Can you improve evaluation section considering the bullet points mentioned in weakness?"

**这是总结性问题**: 要求解决W4-W7的所有评估问题

#### Q2: 理论贡献讨论（非实验需求）
> "Can you discuss what's the new theoretical contribution..."

**性质**: 需要在论文中添加讨论，不需要实验

#### Q3: MoE/LoRA vs Dual Memory讨论（非实验需求）
> "Can you add a section to discuss the fundamental similarity and difference..."

**性质**: 需要在论文中添加讨论，不需要实验

---

## 🔴 Reviewer 4dsu - 详细分析

### Questions（全部是实验需求）

#### Q1: N=1000端到端成本对比 ⭐⭐⭐
> "Could the authors provide an experiment that directly compares the end-to-end wall-clock time or computational cost of REPAIR against baselines for a large-scale (N=1000) editing task?"

**状态**: ✅ **已完成** - Rebuttal论文中有完整的N=1000 throughput分析

#### Q2: 剪枝阈值敏感性分析 ⭐⭐
> "Would it be possible to conduct a sensitivity analysis on the error pruning threshold (τ_prune)..."

**实验需求**: 
- 测试不同τ_prune值
- 分析reliability vs stability的权衡

#### Q3: 假阳性剪枝率追踪 ⭐
> "Could an experiment be designed to track the 'false positive' pruning rate..."

**实验需求**:
- 追踪成功编辑被错误剪枝的比例

---

## 🔴 Reviewer oiCK - 详细分析

### Weaknesses

#### W1: 拼写错误（非实验需求）
> "Intervention" is incorrectly spelled "Intervension"

**性质**: 需要修正拼写，不需要实验

#### W2: 成本分析缺失 ⭐⭐
> "The paper does not provide an analysis of inference latency, parameter increments, or training resource consumption"

**实验需求**:
- Inference latency
- Parameter increments
- Training resource consumption

**状态**: ⚠️ Rebuttal中有部分，可能需要补充

#### W3: 大规模可扩展性说明（非实验需求）
> "it would benefit from a clearer explanation of how REPAIR scales with models of significantly larger sizes"

**性质**: 需要在论文中解释，不一定需要新实验

#### W4: 理论假设过于理想化（非实验需求）
> "Assumption 2 assumes that 'each re-triggering will reduce the error rate by at least a fixed constant δ'"

**性质**: 需要在论文中承认或提供实验证据支持

### Questions

#### Q1: Figure 4消融研究不清晰 ⭐
> "The legend lists four configurations... but the chart appears to show only three comparison curves"

**实验需求**: 
- 需要清晰的KD消融对比
- 确保所有曲线可见

#### Q2: Table 2示例不完整（非实验需求）
> "Table 2 only shows the successful output of REPAIR on one example (row c)"

**性质**: 需要补充案例展示，不需要新实验

---

## 🔴 Reviewer 7U7d - 详细分析

### Weaknesses

#### W1: 排版和格式错误（非实验需求）
> "Excessive typographical and formatting errors"

**性质**: 需要校对论文，不需要实验

#### W2: 固定阈值鲁棒性 ⭐
> "Fixed thresholds may fail under extreme distribution shifts"

**实验需求**: 
- 测试OOD场景下的鲁棒性
- 可选实验

#### W3: 异构批次消融 ⭐⭐⭐
> "An additional experiment or ablation is recommended to examine the effect of heterogeneous batches"

**状态**: ✅ **已完成** - 30.8%局部性退化

#### W4: RECIPE对比 ⭐⭐
> "REPAIR is similarity to RECIPE [1]. A comparative analysis or discussion highlighting key differences would strengthen the paper"

**实验需求**: 
- RECIPE对比实验
- 或详细讨论差异

**状态**: ⚠️ 训练失败，使用范式差异策略

#### W5: 流程描述不详细（非实验需求）
> "Insufficiently detailed process description"

**性质**: 需要改进论文写作，添加流程图

#### W6: 数学符号不一致（非实验需求）
> "Inconsistent mathematical notation and figure labeling"

**性质**: 需要校对论文

### Questions

#### Q1: 同质批次相似性定义 ⭐
> "Would it be possible to provide a more detailed explanation and a clearer definition of what criteria are used to determine similarity"

**实验需求**:
- 相似性度量分析
- 可视化批次内相似度

#### Q2: Closed-Loop对局部性的影响 ⭐
> "When modifying shards that include low-error samples through Closed Loop Feedback, could this process potentially harm the locality"

**实验需求**:
- Closed-Loop的局部性分析
- 可选实验

---

## 📊 实验需求总结（按审稿人评分重新排序）

### 🚨 **最高优先级 - 必须说服4分审稿人**

**G6uc (4分) 和 7U7d (4分) 的要求必须优先满足！**

#### G6uc的核心要求（4分 → 需要翻转）

| # | 实验 | 状态 | 预计时间 | 重要性 |
|---|------|------|---------|--------|
| 1 | **ELDER baseline** | ✅ 可运行 | 2h | ⭐⭐⭐ |
| 2 | **剪枝阈值敏感性** | ❌ 缺失 | 3-4h | ⭐⭐⭐ |
| 3 | **Reasoning locality** | ❌ 缺失 | 4-6h | ⭐⭐ |
| 4 | **标准FT (LoRA)** | ❌ 缺失 | 3-4h | ⭐⭐ |
| 5 | sLKE/LeMOE baseline | ❌ 未实现 | 16-32h | ⭐（可承诺最终版本） |
| 6 | Tool-use locality | ❌ 缺失 | 4-6h | ⭐（可承诺最终版本） |
| 7 | N=5000扩展 | ❌ 缺失 | 10-15h | ⭐（可说明N=1000足够） |

**G6uc最关键的3个实验**（必须完成）:
1. ELDER baseline (2h)
2. 剪枝阈值敏感性 (3-4h)
3. Reasoning locality (4-6h)

#### 7U7d的核心要求（4分 → 需要翻转）

| # | 实验 | 状态 | 预计时间 | 重要性 |
|---|------|------|---------|--------|
| 1 | **异构批次消融** | ✅ **已完成** | - | ⭐⭐⭐ |
| 2 | **RECIPE对比/讨论** | ⚠️ 需说明 | - | ⭐⭐ |
| 3 | **相似性度量分析** | ❌ 缺失 | 2h | ⭐⭐ |
| 4 | OOD鲁棒性 | ❌ 缺失 | 3h | ⭐（可选） |
| 5 | Closed-Loop局部性 | ❌ 缺失 | 2h | ⭐（可选） |

**7U7d最关键的2个实验**（必须完成）:
1. ✅ 异构批次消融（已完成）
2. 相似性度量分析 (2h)

### 🟡 中优先级 - 6分审稿人的要求

#### 4dsu (6分) 的要求

| # | 实验 | 状态 | 预计时间 |
|---|------|------|---------|
| 1 | **N=1000成本对比** | ✅ **已完成** | - |
| 2 | **剪枝阈值敏感性** | ❌ 缺失 | 3-4h |
| 3 | **假阳性剪枝率** | ❌ 缺失 | 2-3h |

#### oiCK (6分) 的要求

| # | 实验 | 状态 | 预计时间 |
|---|------|------|---------|
| 1 | **KD消融（清晰版）** | ⚠️ 需改进 | 2h |
| 2 | **成本分析补充** | ⚠️ 部分完成 | 1-2h |

### 📋 优先级重排（基于评分）

**绝对必须完成**（说服4分审稿人，9-12小时）:
1. ELDER baseline (2h) - G6uc
2. 剪枝阈值敏感性 (3-4h) - G6uc + 4dsu
3. Reasoning locality (4-6h) - G6uc
4. 相似性度量分析 (2h) - 7U7d

**强烈建议完成**（巩固6分审稿人，9-12小时）:
5. 标准FT (LoRA) (3-4h) - G6uc
6. KD消融清晰版 (2h) - oiCK
7. 假阳性剪枝率 (2-3h) - 4dsu
8. 成本分析补充 (1-2h) - oiCK

**可选/可承诺最终版本**:
- sLKE/LeMOE (承诺最终版本)
- Tool-use locality (承诺最终版本)
- N=5000 (说明N=1000足够)
- OOD鲁棒性 (可选)
- Closed-Loop局部性 (可选)

### 📝 非实验需求（论文改进）

| # | 需求 | 审稿人 | 工作量 |
|---|------|--------|--------|
| 17 | 修正拼写错误 | oiCK W1, 7U7d W1 | 1-2h |
| 18 | 改进流程描述 | 7U7d W5 | 2-3h |
| 19 | 统一数学符号 | 7U7d W6 | 2-3h |
| 20 | 理论贡献讨论 | G6uc Q2 | 2-3h |
| 21 | MoE vs Dual Memory讨论 | G6uc Q3 | 2-3h |
| 22 | 理论假设说明 | oiCK W4 | 1-2h |
| 23 | Table 2案例补充 | oiCK Q2 | 1h |
| 24 | 大规模可扩展性说明 | oiCK W3 | 1-2h |

---

## 🎯 最终实验优先级（修正版）

### Phase 1: 必须完成（5-6小时）

**已完成的优势**:
- ✅ N=1000成本对比（4dsu核心要求）
- ✅ 异构批次消融（7U7d核心要求）

**仍需完成**:
1. ELDER baseline @ ZsRE, Hallucination, Temporal (2h)
2. 剪枝阈值敏感性分析 (3-4h)

### Phase 2: 强烈建议（12-15小时）

3. KD消融清晰版本 (2h)
4. 成本分析补充细节 (1-2h)
5. Reasoning locality测试 (4-6h)
6. 标准FT (LoRA)重新运行 (3-4h)
7. 相似性度量分析 (2h)

### Phase 3: 可选补充（20-30小时）

8. 假阳性剪枝率追踪 (2-3h)
9. Tool-use locality (4-6h)
10. OOD鲁棒性 (3h)
11. Closed-Loop局部性 (2h)
12. N=5000扩展 (10-15h)

### Phase 4: 最终版本（需要大量工作）

13. sLKE baseline (8-16h，需实现)
14. LeMOE baseline (8-16h，需实现)

---

## ✅ 最终建议（基于评分调整）

### 🚨 关键策略：必须翻转4分审稿人

**目标**: 将G6uc和7U7d从4分提升到至少5-6分

### 最小可行方案（9-12小时）⭐⭐⭐

**专注于4分审稿人的核心要求**:

1. **ELDER baseline** (2h) - G6uc核心要求
2. **剪枝阈值敏感性** (3-4h) - G6uc核心要求
3. **Reasoning locality** (4-6h) - G6uc核心要求
4. **相似性度量分析** (2h) - 7U7d核心要求

**已完成的优势**:
- ✅ 异构批次消融（7U7d的最重要要求）
- ✅ N=1000成本对比（4dsu的核心要求）

**Rebuttal说明**:
- sLKE/LeMOE: 承诺最终版本
- Tool-use: 承诺最终版本
- N=5000: 说明N=1000足够
- RECIPE: 范式差异说明

### 推荐方案（18-24小时）⭐⭐

**在最小方案基础上，巩固所有审稿人**:

**4分审稿人** (9-12h):
1. ELDER baseline (2h)
2. 剪枝阈值敏感性 (3-4h)
3. Reasoning locality (4-6h)
4. 相似性度量分析 (2h)

**6分审稿人** (9-12h):
5. 标准FT (LoRA) (3-4h) - 也满足G6uc
6. KD消融清晰版 (2h) - oiCK
7. 假阳性剪枝率 (2-3h) - 4dsu
8. 成本分析补充 (1-2h) - oiCK

### 完整方案（30-40小时）

**包含所有实验 + 论文改进**:
- 所有上述实验 (18-24h)
- 论文改进 (10-15h)
  - 修正拼写错误
  - 改进流程描述
  - 统一数学符号
  - 添加理论讨论

### 时间分配建议

**如果只有10小时**:
- 专注于4分审稿人的最关键实验
- ELDER + 阈值敏感性 + 相似性分析

**如果有15小时**:
- 完成4分审稿人的所有核心要求
- 包括Reasoning locality

**如果有20-25小时**:
- 完成推荐方案
- 同时满足4分和6分审稿人

### 预期效果

**最小方案**:
- G6uc: 4分 → 5-6分（满足核心要求）
- 7U7d: 4分 → 6分（异构批次已完成+相似性分析）
- 4dsu: 6分 → 保持（N=1000已完成）
- oiCK: 6分 → 保持

**推荐方案**:
- G6uc: 4分 → 6-7分（满足所有主要要求）
- 7U7d: 4分 → 6-7分（满足所有要求）
- 4dsu: 6分 → 7分（满足所有要求）
- oiCK: 6分 → 7分（满足所有要求）

### 关键优势

✅ **已完成的重要实验**:
- N=1000成本对比（4dsu最重要）
- 异构批次消融（7U7d最重要）

✅ **只需9-12小时即可翻转4分审稿人**

✅ **18-24小时可以满足所有审稿人的核心要求**
