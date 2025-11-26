#!/usr/bin/env python3
"""
Cost Analysis Script for REPAIR vs WISE
Tracks: inference latency, GPU memory, training time, parameter increments
"""
import os
import sys
import json
import argparse
import time
import torch
import subprocess
from pathlib import Path

sys.path.append('..')
from easyeditor import (
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip().split('\n')[0])
    except:
        return 0


def get_model_params(model):
    """Calculate total parameters in millions"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_side_memory_size(editor):
    """Calculate Side Memory size in MB"""
    try:
        adapter = editor.model.get_adapter_layer() if hasattr(editor.model, 'get_adapter_layer') else None
        if adapter and hasattr(adapter, 'memory_weight'):
            total_params = sum(w.numel() for w in adapter.memory_weight)
            size_mb = total_params * 4 / 1024 / 1024  # float32
            return size_mb, len(adapter.memory_weight)
        return 0, 0
    except:
        return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=False, type=str, default='WISE')
    parser.add_argument('--hparams_dir', required=False, type=str, default='../hparams/WISE/llama-3-8b.yaml')
    parser.add_argument('--data_dir', required=False, type=str, default='../data/wise')
    parser.add_argument('--data_type', required=False, type=str, default='ZsRE')
    parser.add_argument('--output_dir', default='./cost_analysis', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
    parser.add_argument('--sequential_edit', action="store_true", default=True)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    K = args.ds_size
    edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
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
    
    print(f"\n{'='*60}")
    print(f"Cost Analysis: {args.editing_method} @ N={args.ds_size}")
    print(f"{'='*60}\n")
    
    # Track initial GPU memory
    gpu_mem_before = get_gpu_memory()
    print(f"Initial GPU Memory: {gpu_mem_before} MB")
    
    # Create editor and track time
    print("\nInitializing editor...")
    init_start = time.time()
    editor = BaseEditor.from_hparams(hparams)
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.2f}s")
    
    # Track model parameters
    base_params = get_model_params(editor.model.model if hasattr(editor.model, 'model') else editor.model)
    print(f"Base model parameters: {base_params:.2f}M")
    
    # Track editing time
    print(f"\nStarting editing on {K} samples...")
    edit_start = time.time()
    gpu_mem_peak = gpu_mem_before
    
    # Run editing
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric='token em'
    )
    
    edit_time = time.time() - edit_start
    gpu_mem_after = get_gpu_memory()
    gpu_mem_peak = max(gpu_mem_peak, gpu_mem_after)
    
    print(f"\nEditing completed in {edit_time:.2f}s")
    print(f"Average time per edit: {edit_time/K*1000:.2f}ms")
    
    # Calculate Side Memory size
    side_memory_mb, num_shards = get_side_memory_size(editor)
    print(f"Side Memory: {side_memory_mb:.2f} MB ({num_shards} shards)")
    
    # Track inference latency
    print("\nMeasuring inference latency...")
    test_prompts = prompts[:min(10, len(prompts))]
    
    inference_times = []
    for prompt in test_prompts:
        start = time.time()
        _ = edited_model.generate(
            **editor.tok([prompt], return_tensors='pt').to(editor.model.device),
            max_new_tokens=20
        )
        inference_times.append((time.time() - start) * 1000)  # ms
    
    avg_inference_latency = sum(inference_times) / len(inference_times)
    print(f"Average inference latency: {avg_inference_latency:.2f}ms")
    
    # Compile cost analysis results
    cost_analysis = {
        'method': args.editing_method,
        'dataset_size': K,
        'initialization_time_s': round(init_time, 2),
        'total_editing_time_s': round(edit_time, 2),
        'avg_edit_time_ms': round(edit_time/K*1000, 2),
        'avg_inference_latency_ms': round(avg_inference_latency, 2),
        'base_params_M': round(base_params, 2),
        'side_memory_MB': round(side_memory_mb, 2),
        'num_memory_shards': num_shards,
        'gpu_memory_before_MB': gpu_mem_before,
        'gpu_memory_after_MB': gpu_mem_after,
        'gpu_memory_peak_MB': gpu_mem_peak,
        'gpu_memory_increase_MB': gpu_mem_after - gpu_mem_before,
    }
    
    # Save cost analysis
    cost_file = os.path.join(args.output_dir, f'{args.editing_method}_N{K}_cost.json')
    with open(cost_file, 'w') as f:
        json.dump(cost_analysis, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Cost Analysis Summary:")
    print(f"{'='*60}")
    for key, value in cost_analysis.items():
        print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")
    
    print(f"Cost analysis saved to: {cost_file}")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, f'{args.editing_method}_N{K}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if len(metrics) > 0:
        print("\nEditing Metrics:")
        summary_metrics(metrics)
