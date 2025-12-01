#!/usr/bin/env python3
"""
运行MELO baseline实验
MELO是ELDER的基础方法，都是MoE-based lifelong editing
"""

import os
import sys
import json
import argparse
sys.path.append('..')

from easyeditor import (
    MELOHyperParams,
    BaseEditor,
    summary_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=False, type=str, default='../data/wise')
    parser.add_argument('--data_type', required=False, type=str,
                        choices=['ZsRE', 'Temporal', 'Hallucination'], default='ZsRE')
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
    parser.add_argument('--sequential_edit', action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Load hyperparameters
    hparams = MELOHyperParams.from_hparams(args.hparams_dir)
    
    # Load data
    if args.data_type == 'ZsRE':
        data_file = os.path.join(args.data_dir, 'ZsRE', 'zsre_mend_edit.json')
    elif args.data_type == 'Temporal':
        data_file = os.path.join(args.data_dir, 'Temporal', 'temporal_edit.json')
    elif args.data_type == 'Hallucination':
        data_file = os.path.join(args.data_dir, 'Hallucination', 'hallucination_edit.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Take first K samples
    K = args.ds_size
    prompts = [data[i]['src'] for i in range(K)]
    rephrase_prompts = [data[i]['rephrase'] for i in range(K)]
    target_new = [data[i]['alt'] for i in range(K)]
    locality_prompts = [data[i]['loc'] for i in range(K)]
    locality_ans = [data[i]['loc_ans'] for i in range(K)]
    subject = [data[i]['subject'] for i in range(K)]
    
    # Create editor
    editor = BaseEditor.from_hparams(hparams)
    
    # Run editing
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        locality_inputs={
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            }
        },
        sequential_edit=args.sequential_edit
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'MELO_{args.data_type}_N{args.ds_size}.json'
    )
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    summary = summary_metrics(metrics)
    print("\n" + "="*60)
    print("MELO Baseline Results")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
