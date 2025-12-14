#!/usr/bin/env python3
"""
提取和分析ELDER实验结果
"""
import pickle
import json
import numpy as np
from pathlib import Path

# ELDER结果文件
elder_result_file = '/root/REPAIR/external_baselines/MELO/melo/outputs/2025-12-01_03-55-39_8848091074/log.pkl'

# 加载ELDER结果
print("=== 加载ELDER结果 ===")
with open(elder_result_file, 'rb') as f:
    elder_results = pickle.load(f)

print(f"结果文件键: {elder_results.keys()}")
print()

# 提取关键指标
print("=== ELDER ZsRE 结果分析 ===")
print()

# 检查结果结构
for key in elder_results.keys():
    print(f"{key}: {type(elder_results[key])}")
    if isinstance(elder_results[key], dict):
        print(f"  子键: {list(elder_results[key].keys())[:5]}...")
    elif isinstance(elder_results[key], list):
        print(f"  长度: {len(elder_results[key])}")
print()

# 提取性能指标
if 'all_UP' in elder_results:
    print("Upstream Performance (保持原有知识):")
    for n_edits in sorted(elder_results['all_UP'].keys()):
        print(f"  After {n_edits} edits: {elder_results['all_UP'][n_edits]}")
    print()

if 'all_HIS' in elder_results:
    print("History Performance (历史编辑保持):")
    for n_edits in sorted(elder_results['all_HIS'].keys()):
        print(f"  After {n_edits} edits: {elder_results['all_HIS'][n_edits]}")
    print()

if 'all_HOLDOUT' in elder_results:
    print("Holdout Performance:")
    for n_edits in sorted(elder_results['all_HOLDOUT'].keys()):
        print(f"  After {n_edits} edits: {elder_results['all_HOLDOUT'][n_edits]}")
    print()

if 'all_edit_time' in elder_results:
    print("Edit Time:")
    for n_edits in sorted(elder_results['all_edit_time'].keys()):
        print(f"  After {n_edits} edits: {elder_results['all_edit_time'][n_edits]:.2f}s")
    print()

# 保存分析结果
output_dir = Path('/root/REPAIR/rebuttal_experiments_final/elder_analysis')
output_dir.mkdir(exist_ok=True)

# 保存为JSON（转换numpy类型）
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

elder_summary = {
    'upstream': convert_to_serializable(elder_results.get('all_UP', {})),
    'history': convert_to_serializable(elder_results.get('all_HIS', {})),
    'holdout': convert_to_serializable(elder_results.get('all_HOLDOUT', {})),
    'edit_time': convert_to_serializable(elder_results.get('all_edit_time', {})),
}

with open(output_dir / 'elder_zsre_summary.json', 'w') as f:
    json.dump(elder_summary, f, indent=2)

print(f"结果已保存到: {output_dir / 'elder_zsre_summary.json'}")
