#!/bin/bash
# ICLR Rebuttal实验执行脚本
# 基于审稿人评分优化的优先级顺序

set -e

echo "=========================================="
echo "ICLR Rebuttal实验 - 优先级执行"
echo "=========================================="
echo ""
echo "目标: 翻转4分审稿人 (G6uc, 7U7d)"
echo ""

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="rebuttal_experiments_final/${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}/{elder,threshold,reasoning,similarity,logs}

echo "结果将保存到: ${RESULTS_DIR}"
echo ""

# ============================================
# Phase 1: 4分审稿人的核心实验 (9-12小时)
# ============================================

echo "=========================================="
echo "Phase 1: 4分审稿人核心实验"
echo "=========================================="

# 实验1: ELDER baseline (2小时)
echo ""
echo "[1/4] 运行 ELDER baseline @ ZsRE, Hallucination, Temporal..."
echo "预计时间: 2小时"
echo "审稿人: G6uc (4分)"

for dataset in ZsRE Hallucination Temporal; do
    echo "  - Running ELDER on ${dataset}..."
    uv run examples/run_wise_editing.py \
        --editing_method ELDER \
        --hparams_dir hparams/ELDER/qwen2.5-7b.yaml \
        --data_dir data/wise \
        --data_type ${dataset} \
        --ds_size 100 \
        --output_file "${RESULTS_DIR}/elder/ELDER_${dataset}_N100.json" \
        2>&1 | tee "${RESULTS_DIR}/logs/elder_${dataset}.log"
    echo "  ✓ Completed ELDER on ${dataset}"
done

echo "✓ ELDER baseline完成"

# 实验2: 剪枝阈值敏感性分析 (3-4小时)
echo ""
echo "[2/4] 运行剪枝阈值敏感性分析..."
echo "预计时间: 3-4小时"
echo "审稿人: G6uc (4分), 4dsu (6分)"

# 创建敏感性分析脚本
cat > ${RESULTS_DIR}/threshold/run_sensitivity.py << 'PYEOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from examples.run_wise_editing import main as run_editing
import json
from pathlib import Path

# 测试不同阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for tau in thresholds:
    print(f"\n{'='*60}")
    print(f"Testing threshold: {tau}")
    print(f"{'='*60}")
    
    # 修改hparams中的阈值
    # 这里需要根据实际的hparams结构调整
    # 运行实验并收集结果
    
    # 占位符 - 实际需要实现
    result = {
        'threshold': tau,
        'rewrite_acc': 0.0,  # 需要从实验结果中提取
        'locality': 0.0,
        'num_pruned': 0
    }
    results.append(result)

# 保存结果
output_file = Path('threshold_sensitivity_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")
PYEOF

echo "  注意: 阈值敏感性分析需要修改WISE_new.py以支持动态阈值"
echo "  建议: 手动运行不同阈值配置的实验"

# 实验3: Reasoning locality测试 (4-6小时)
echo ""
echo "[3/4] 运行Reasoning locality测试..."
echo "预计时间: 4-6小时"
echo "审稿人: G6uc (4分)"

cat > ${RESULTS_DIR}/reasoning/run_reasoning_locality.py << 'PYEOF'
"""
Reasoning Locality测试

测试在ZsRE编辑后，模型的reasoning能力是否保持
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 1. 加载模型
# 2. 测试编辑前的reasoning能力 (GSM8K, MMLU)
# 3. 进行N=100次ZsRE编辑
# 4. 测试编辑后的reasoning能力
# 5. 计算locality = (after - before) / before

print("Reasoning locality测试需要:")
print("1. GSM8K数据集")
print("2. MMLU数据集")
print("3. 评估脚本")
print("")
print("建议使用lm-evaluation-harness进行评估")
PYEOF

echo "  注意: Reasoning locality需要额外的数据集和评估框架"

# 实验4: 相似性度量分析 (2小时)
echo ""
echo "[4/4] 运行相似性度量分析..."
echo "预计时间: 2小时"
echo "审稿人: 7U7d (4分)"

cat > ${RESULTS_DIR}/similarity/analyze_batch_similarity.py << 'PYEOF'
"""
批次相似性分析

分析同质批次vs异构批次的特征相似度
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json

def extract_features(model, inputs):
    """提取模型的隐藏层特征"""
    # 需要实现特征提取逻辑
    pass

def compute_similarity_matrix(features):
    """计算相似度矩阵"""
    return cosine_similarity(features)

def plot_similarity_distribution(sim_homo, sim_hetero, output_path):
    """可视化相似度分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(sim_homo.flatten(), bins=50, alpha=0.5, label='Homogeneous')
    plt.hist(sim_hetero.flatten(), bins=50, alpha=0.5, label='Heterogeneous')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Batch Similarity Distribution')
    plt.savefig(output_path)
    plt.close()

def plot_tsne(features_homo, features_hetero, output_path):
    """t-SNE可视化"""
    # 合并特征
    all_features = np.vstack([features_homo, features_hetero])
    labels = ['Homo'] * len(features_homo) + ['Hetero'] * len(features_hetero)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(all_features)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    for label in ['Homo', 'Hetero']:
        mask = [l == label for l in labels]
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   label=label, alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of Batch Features')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    print("相似性分析脚本")
    print("需要实现:")
    print("1. 特征提取")
    print("2. 相似度计算")
    print("3. 可视化")
PYEOF

echo "  注意: 相似性分析需要访问模型的隐藏层表示"

echo ""
echo "=========================================="
echo "Phase 1完成"
echo "=========================================="
echo ""
echo "已完成4分审稿人的核心实验框架"
echo ""
echo "下一步:"
echo "1. 完善阈值敏感性分析的实现"
echo "2. 准备Reasoning locality的数据集"
echo "3. 实现相似性分析的特征提取"
echo ""
echo "预计总时间: 9-12小时"
echo ""

# ============================================
# Phase 2: 6分审稿人的实验 (可选)
# ============================================

echo "=========================================="
echo "Phase 2: 6分审稿人实验 (可选)"
echo "=========================================="
echo ""
echo "如果时间允许，运行以下实验:"
echo "1. 标准FT (LoRA) - 3-4小时"
echo "2. KD消融清晰版 - 2小时"
echo "3. 假阳性剪枝率 - 2-3小时"
echo "4. 成本分析补充 - 1-2小时"
echo ""

echo "=========================================="
echo "实验脚本生成完成"
echo "=========================================="
echo ""
echo "结果目录: ${RESULTS_DIR}"
echo ""
echo "建议执行顺序:"
echo "1. ELDER baseline (必须)"
echo "2. 剪枝阈值敏感性 (必须)"
echo "3. Reasoning locality (必须)"
echo "4. 相似性分析 (必须)"
echo ""
echo "总预计时间: 9-12小时"
