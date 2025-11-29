# ICLR Rebuttal实验快速开始指南

## 🎯 目标

翻转4分审稿人（G6uc和7U7d）的评分，从4分提升到6分以上。

## ✅ 已完成的实验

- ✅ **N=1000成本对比** - 4dsu的核心要求
- ✅ **异构批次消融** - 7U7d的最重要要求

## 🚨 必须完成的实验（9-12小时）

### 1. ELDER Baseline (2小时)

```bash
# 运行ELDER在三个数据集上
for dataset in ZsRE Hallucination Temporal; do
    uv run examples/run_wise_editing.py \
        --editing_method ELDER \
        --hparams_dir hparams/ELDER/qwen2.5-7b.yaml \
        --data_type ${dataset} \
        --ds_size 100
done
```

**审稿人**: G6uc (4分)

### 2. 剪枝阈值敏感性分析 (3-4小时)

需要修改`WISE_new.py`以支持动态阈值，然后测试不同τ_prune值。

**审稿人**: G6uc (4分), 4dsu (6分)

### 3. Reasoning Locality测试 (4-6小时)

```bash
# 1. 测试编辑前的reasoning能力
# 2. 进行N=100次ZsRE编辑
# 3. 测试编辑后的reasoning能力
# 4. 计算locality保持率
```

需要GSM8K和MMLU数据集。

**审稿人**: G6uc (4分)

### 4. 相似性度量分析 (2小时)

分析同质批次vs异构批次的特征相似度。

**审稿人**: 7U7d (4分)

## 📋 完整文档

- **详细分析**: `ICLR_REBUTTAL_EXPERIMENT_PLAN.md`
- **执行脚本**: `run_rebuttal_priority_experiments.sh`

## 🔧 并行执行建议

如果有多台机器，可以并行运行：

**机器1**: ELDER baseline + 相似性分析 (4小时)
**机器2**: 剪枝阈值敏感性 (3-4小时)
**机器3**: Reasoning locality (4-6小时)

## 📊 预期效果

完成这4个实验后：
- G6uc: 4分 → 5-6分
- 7U7d: 4分 → 6分
- 论文接受概率显著提升

## 🎯 Rebuttal策略

对于无法完成的实验（sLKE, LeMOE, N=5000）：
- 承诺在最终版本补充
- 说明实现复杂度和时间限制
- 强调已完成实验的充分性
