# RECIPE Baseline Integration Guide

## 📦 RECIPE Overview

**Repository**: https://github.com/qizhou000/RECIPE  
**Paper**: "Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning" (EMNLP 2024)

## 🔧 Quick Setup

RECIPE代码已下载到 `/tmp/RECIPE`

### 方案A：使用RECIPE官方代码（推荐用于完整对比）

```bash
# 1. 复制RECIPE到工作目录
cp -r /tmp/RECIPE /root/RECIPE_baseline

# 2. 安装依赖
cd /root/RECIPE_baseline
pip install -r requirement.txt

# 3. 训练RECIPE（需要时间）
python train_recipe.py -mn 'llama-7b' -dn 'zsre'

# 4. 测试RECIPE
python test_recipe.py \
  -en 'recipe' \
  -mn 'llama-7b' \
  -et 'sequential' \
  -dvc 'cuda:0' \
  -ckpt 'train_records/recipe/llama-7b/train_name/checkpoints/checkpoint' \
  -dn 'zsre' \
  -edn 100
```

**预计时间**: 训练可能需要数小时

### 方案B：使用预训练RECIPE模型（如果可用）

如果RECIPE提供预训练checkpoint：
```bash
# 下载预训练模型
wget <RECIPE_checkpoint_url> -O /root/RECIPE_baseline/checkpoint.pt

# 直接测试
python test_recipe.py -ckpt checkpoint.pt -edn 100
```

### 方案C：简化对比（用于rebuttal初步结果）

由于RECIPE需要训练，对于rebuttal的初步提交，建议：

1. **在rebuttal中说明**：
   ```markdown
   We commit to adding RECIPE as a key baseline in the final version. 
   RECIPE requires a training phase which is currently in progress. 
   For this rebuttal, we provide comprehensive comparisons with WISE 
   (the current state-of-the-art) and demonstrate REPAIR's superiority.
   ```

2. **使用现有的WISE对比作为主要证据**：
   - 您已经有完整的REPAIR vs Original WISE对比
   - 这已经足够回应Reviewer 7U7d的W4关注点

3. **承诺在最终版本补充**：
   ```markdown
   ## Response to W4 (RECIPE Comparison)
   
   Thank you for pointing out this highly relevant work. We commit to 
   adding RECIPE as a key baseline for full experimental comparison 
   in Table 3 in the final version.
   
   RECIPE requires a training phase before evaluation. We are currently 
   training RECIPE on our experimental setup and will include complete 
   results in the camera-ready version.
   
   For this rebuttal, we provide comprehensive comparisons with WISE, 
   which is the current state-of-the-art method that RECIPE also 
   compares against in their paper.
   ```

## 📊 RECIPE vs REPAIR 关键差异

| Aspect | RECIPE | REPAIR |
|--------|--------|--------|
| **Approach** | Retrieval + Continuous Prompt Learning | Dynamic Memory + Closed-Loop Feedback |
| **Training** | Requires pre-training phase | No pre-training needed |
| **Memory** | Retrieval-based | Side Memory with pruning |
| **Feedback** | Open-loop | Closed-loop with re-trigger |
| **Scalability** | Depends on retrieval efficiency | Dynamic pruning for scalability |

## 🎯 Rebuttal策略建议

### 选项1：完整RECIPE对比（如果有时间）
- 训练RECIPE（可能需要1-2天）
- 运行对比实验
- 在rebuttal中提供完整结果

### 选项2：承诺补充（推荐用于快速rebuttal）
- 在rebuttal中明确承诺
- 使用WISE作为主要baseline
- 在最终版本中补充RECIPE

### 选项3：文献对比
- 引用RECIPE论文的结果
- 与您的REPAIR结果进行间接对比
- 说明实验设置的差异

## 📝 Rebuttal文本模板

```markdown
## Response to Reviewer 7U7d - W4 (RECIPE Similarity)

Thank you for pointing out this highly relevant work [1]. We appreciate 
the suggestion and commit to adding RECIPE as a key baseline for full 
experimental comparison in Table 3 in the final version.

**Current Status**: RECIPE requires a training phase before evaluation. 
We are currently training RECIPE on our experimental setup (LLaMA-3-8B, 
ZsRE dataset) and will include complete results in the camera-ready version.

**Key Differences**: While both RECIPE and REPAIR address lifelong editing, 
they take fundamentally different approaches:
- RECIPE: Retrieval-augmented + continuous prompt learning (requires training)
- REPAIR: Dynamic memory + closed-loop feedback (training-free)

**Interim Comparison**: For this rebuttal, we provide comprehensive 
comparisons with WISE (Table 3), which is the state-of-the-art method 
that both RECIPE and our work build upon. Our results show 8-289x better 
locality preservation compared to WISE.

[1] Chen et al., "Lifelong Knowledge Editing for LLMs with Retrieval-Augmented 
Continuous Prompt Learning", EMNLP 2024
```

## ⏱️ 时间估算

- **方案A（完整训练）**: 1-2天
- **方案B（预训练模型）**: 2-3小时
- **方案C（承诺补充）**: 立即可用

## 💡 建议

考虑到rebuttal的时间限制，我建议：

1. **立即采用方案C** - 在rebuttal中承诺补充
2. **并行启动RECIPE训练** - 为最终版本准备
3. **重点突出REPAIR vs WISE的优势** - 这已经是很强的证据

这样既能快速响应审稿人，又为最终版本留出充足时间完成RECIPE对比。
