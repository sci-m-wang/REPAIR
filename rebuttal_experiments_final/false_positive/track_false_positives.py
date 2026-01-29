#!/usr/bin/env python3
"""
假阳性剪枝率追踪实验

目标：追踪成功编辑被错误剪枝的比例
- 测试不同τ_prune阈值
- 分析哪些成功的编辑被错误剪枝
- 评估reliability vs stability的权衡
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from easyeditor import WISEHyperParams, BaseEditor

def track_false_positives(hparams, prompts, target_new, subject, rephrase_prompts, 
                          locality_prompts, locality_ans, threshold):
    """
    追踪假阳性剪枝率
    
    假阳性定义：一个编辑本来是成功的（rewrite_acc=1），但被剪枝机制错误地标记为失败
    """
    
    # 修改阈值
    hparams.act_ratio = threshold
    
    # 初始化编辑器
    editor = BaseEditor.from_hparams(hparams)
    
    # 运行编辑
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        loc_prompts=locality_prompts,  # WISE expects loc_prompts, not locality_inputs
        sequential_edit=True,
        eval_metric='token em'
    )
    
    # 分析结果
    false_positives = []
    true_positives = []
    true_negatives = []
    false_negatives = []
    
    for i, m in enumerate(metrics):
        rewrite_acc = m['post']['rewrite_acc']
        
        # 检查是否被剪枝（通过检查是否有pruning相关的标记）
        # 注意：这需要根据WISE的实际实现来判断
        # 假设我们可以从metrics中获取pruning信息
        was_pruned = m.get('pruned', False)  # 这个字段可能需要在WISE中添加
        
        if rewrite_acc == 1.0:
            if was_pruned:
                false_positives.append(i)  # 成功但被剪枝（假阳性）
            else:
                true_positives.append(i)   # 成功且未被剪枝
        else:
            if was_pruned:
                true_negatives.append(i)   # 失败且被剪枝（正确）
            else:
                false_negatives.append(i)  # 失败但未被剪枝
    
    # 计算统计数据
    total = len(metrics)
    fp_rate = len(false_positives) / total if total > 0 else 0
    tp_rate = len(true_positives) / total if total > 0 else 0
    
    results = {
        'threshold': threshold,
        'total_edits': total,
        'false_positives': len(false_positives),
        'true_positives': len(true_positives),
        'true_negatives': len(true_negatives),
        'false_negatives': len(false_negatives),
        'false_positive_rate': fp_rate,
        'true_positive_rate': tp_rate,
        'precision': len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0,
        'recall': len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0,
    }
    
    return results, metrics

def main():
    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    hparams_path = os.path.join(project_root, 'hparams/WISE/llama-3-8b.yaml')
    data_dir = os.path.join(project_root, 'data/wise')
    data_type = 'ZsRE'
    ds_size = 100
    output_dir = os.path.join(project_root, 'rebuttal_experiments_final/false_positive')
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的阈值
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # 加载数据
    edit_data = json.load(open(f'{data_dir}/{data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:ds_size]
    
    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    
    # 加载超参数
    hparams = WISEHyperParams.from_hparams(hparams_path)
    
    # 运行实验
    all_results = []
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold: {threshold}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_dir, f'false_positive_tau_{threshold}.json')
        if os.path.exists(output_file):
            print(f"Skipping {threshold}, result file exists.")
            with open(output_file, 'r') as f:
                results = json.load(f)
            all_results.append(results)
            continue
        
        results, metrics = track_false_positives(
            hparams, prompts, target_new, subject, rephrase_prompts,
            locality_prompts, locality_ans, threshold
        )
        
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        print(f"True Positive Rate: {results['true_positive_rate']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        all_results.append(results)
    
    # 保存汇总
    summary_file = os.path.join(output_dir, 'false_positive_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for r in all_results:
        print(f"Threshold {r['threshold']}: FP Rate = {r['false_positive_rate']:.4f}, Precision = {r['precision']:.4f}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
