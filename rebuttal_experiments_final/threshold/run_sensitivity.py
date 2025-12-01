import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/workspace/REPAIR')

from easyeditor import WISEHyperParams, BaseEditor, summary_metrics

def main():
    # Setup
    hparams_path = 'hparams/WISE/qwen2.5-7b.yaml' # Using the Qwen config as base
    data_dir = 'data/wise'
    data_type = 'ZsRE'
    ds_size = 100
    output_dir = 'rebuttal_experiments_final/threshold'
    os.makedirs(output_dir, exist_ok=True)

    # Thresholds to test (act_ratio)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_summary = []

    # Load Data (ZsRE)
    K = ds_size
    edit_data = json.load(open(f'{data_dir}/{data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{data_dir}/{data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
    
    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    # Loop through thresholds
    for act_ratio in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold (act_ratio): {act_ratio}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_dir, f'sensitivity_tau_{act_ratio}.json')
        if os.path.exists(output_file):
            print(f"Skipping {act_ratio}, result file exists.")
            continue

        # Load and modify hparams
        hparams = WISEHyperParams.from_hparams(hparams_path)
        hparams.model_name = "/root/workspace/share/LLM-Research/Meta-Llama-3-8B-Instruct" # Override model path
        hparams.act_ratio = act_ratio # Modify the threshold
        hparams.device = 0 # Force GPU 0

        # Initialize Editor
        editor = BaseEditor.from_hparams(hparams)
        
        # Run Edit
        metrics, _, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=locality_prompts, # Note: BaseEditor.edit signature might vary slightly, checking usage in run_wise_editing.py
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=True,
            eval_metric='token em'
        )

        # Calculate Summary Metrics
        rewrite_acc = np.mean([m['post']['rewrite_acc'] for m in metrics])
        locality_acc = np.mean([m['post']['locality']['neighborhood_acc'] for m in metrics])
        
        print(f"Threshold: {act_ratio} | Rewrite Acc: {rewrite_acc:.4f} | Locality: {locality_acc:.4f}")

        # Save individual result
        output_file = os.path.join(output_dir, f'sensitivity_tau_{act_ratio}.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        results_summary.append({
            'threshold': act_ratio,
            'rewrite_acc': rewrite_acc,
            'locality': locality_acc
        })

    # Save Summary
    summary_file = os.path.join(output_dir, 'sensitivity_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nSummary saved to {summary_file}")

if __name__ == "__main__":
    main()
