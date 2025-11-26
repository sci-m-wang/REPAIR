#!/usr/bin/env python3
"""
Heterogeneous Batch Ablation Study
Tests the impact of editing heterogeneous vs homogeneous batches
"""
import os
import sys
import json
import argparse
import random

sys.path.append('..')
from easyeditor import (
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)


def create_heterogeneous_batches(data, batch_size=10):
    """
    Create heterogeneous batches by randomly shuffling data
    (simulates worst-case scenario where similar edits are not grouped)
    """
    shuffled = data.copy()
    random.shuffle(shuffled)
    return [shuffled[i:i+batch_size] for i in range(0, len(shuffled), batch_size)]


def create_homogeneous_batches(data, batch_size=10):
    """
    Create homogeneous batches by keeping data in original order
    (assumes original data has some natural grouping/similarity)
    """
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=False, type=str, default='../hparams/WISE/llama-3-8b.yaml')
    parser.add_argument('--data_dir', required=False, type=str, default='../data/wise')
    parser.add_argument('--output_dir', default='./ablation_results', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
    parser.add_argument('--heterogeneous', action='store_true', help='Use heterogeneous batches (random shuffle)')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    K = args.ds_size
    edit_data = json.load(open(f'{args.data_dir}/ZsRE/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{args.data_dir}/ZsRE/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
    
    # Prepare batches
    batch_type = "heterogeneous" if args.heterogeneous else "homogeneous"
    print(f"\n{'='*60}")
    print(f"Ablation Study: {batch_type.upper()} Batches @ N={K}")
    print(f"{'='*60}\n")
    
    if args.heterogeneous:
        print("Creating HETEROGENEOUS batches (random shuffle)...")
        # Shuffle the data to create heterogeneous batches
        combined = list(zip(edit_data, loc_data))
        random.shuffle(combined)
        edit_data, loc_data = zip(*combined)
        edit_data = list(edit_data)
        loc_data = list(loc_data)
    else:
        print("Using HOMOGENEOUS batches (original order)...")
    
    # Prepare inputs
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    # Load hyperparameters
    hparams = WISEHyperParams.from_hparams(f'{args.hparams_dir}')
    
    # Create editor
    print("Initializing editor...")
    editor = BaseEditor.from_hparams(hparams)
    
    # Run editing
    print(f"\nStarting editing on {K} samples...")
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=True,
        eval_metric='token em'
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, f'REPAIR_{batch_type}_N{K}.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    if len(metrics) > 0:
        print(f"\n{batch_type.upper()} Batch Results:")
        summary_metrics(metrics)
        
        # Calculate aggregate metrics
        from statistics import mean
        
        post_rewrite = []
        post_rephrase = []
        locality = []
        
        for case in metrics:
            post_rewrite.extend(case['post']['rewrite_acc'])
            if 'rephrase_acc' in case['post']:
                post_rephrase.extend(case['post']['rephrase_acc'])
            locality.extend(case['post']['locality']['neighborhood_acc'])
        
        print(f"\nAggregate Metrics ({batch_type}):")
        print(f"  Rewrite Accuracy:  {mean(post_rewrite):.3f}")
        if post_rephrase:
            print(f"  Rephrase Accuracy: {mean(post_rephrase):.3f}")
        print(f"  Locality:          {mean(locality):.3f}")
