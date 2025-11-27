
import os
import sys
import json
import argparse
import time
import torch
import psutil
from easyeditor import (
    FTHyperParams,
    WISEHyperParams,
    GraceHyperParams,
    BaseEditor,
    summary_metrics,
)

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max GPU Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, type=str, choices=['WISE', 'FT', 'GRACE'])
    parser.add_argument('--hparams', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_dir', default='./outputs_comparison', type=str)
    parser.add_argument('--ds_size', default=10, type=int)
    parser.add_argument('--data_type', required=True, type=str, choices=['custom', 'ZsRE', 'hallucination', 'temporal', 'mental_health', 'medical'])
    parser.add_argument('--batch_sft', action='store_true', help='Run in batch mode (non-sequential)')
    
    args = parser.parse_args()

    print(f"Running {args.method} with hparams {args.hparams} on {args.data_type} (Batch: {args.batch_sft})")
    print_gpu_memory()

    if args.method == 'FT':
        editing_hparams = FTHyperParams
    elif args.method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.method == 'GRACE':
        editing_hparams = GraceHyperParams
    else:
        raise NotImplementedError

    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)[:args.ds_size]
    
    requests = []
    eval_metric = 'token_em' # Default

    if args.data_type in ['custom', 'mental_health', 'medical']:
        for d in data:
            req = {
                'prompt': d['prompt'],
                'target_new': d['target_new'],
                'ground_truth': d.get('ground_truth', '<|endoftext|>'),
                'loc_prompt': d.get('loc_prompt', 'What is the capital of France?'),
                'loc_ans': d.get('loc_ans', 'Paris'),
                'locality': {
                    'neighborhood': {
                        'prompt': [d.get('loc_prompt', 'What is the capital of France?')],
                        'ground_truth': [d.get('loc_ans', 'Paris')]
                    }
                }
            }
            requests.append(req)
    elif args.data_type == 'ZsRE':
        eval_metric = 'token em'
        for d in data:
            req = {
                'prompt': d['src'],
                'target_new': d['alt'],
                'ground_truth': d.get('answers', ['<|endoftext|>'])[0],
                'rephrase_prompt': d['rephrase'],
                'locality': {
                    'neighborhood': {
                        'prompt': [d['loc']],
                        'ground_truth': [d['loc_ans']]
                    }
                }
            }
            requests.append(req)
    elif args.data_type == 'hallucination':
        eval_metric = 'ppl'
        for d in data:
            req = {
                'prompt': d['prompt'],
                'target_new': d['target_new'],
                'ground_truth': d['target_new'],
                'loc_prompt': d['locality_prompt'],
                'loc_ans': d['locality_ground_truth'],
                'locality': {
                    'neighborhood': {
                        'prompt': [d['locality_prompt']],
                        'ground_truth': [d['locality_ground_truth']]
                    }
                }
            }
            requests.append(req)
    elif args.data_type == 'temporal':
        eval_metric = 'ood_ppl' # or ppl
        for d in data:
            req = {
                'prompt': d['prompt'],
                'target_new': d['target_new'],
                'ground_truth': d['target_new'],
                'loc_prompt': d['locality_prompt'],
                'loc_ans': d['locality_ground_truth'],
                'locality': {
                    'neighborhood': {
                        'prompt': [d['locality_prompt']],
                        'ground_truth': [d['locality_ground_truth']]
                    }
                }
            }
            requests.append(req)

    # Load hparams
    hparams = editing_hparams.from_hparams(args.hparams)

    # Initialize Editor
    print("Initializing Editor...")
    start_init = time.time()
    editor = BaseEditor.from_hparams(hparams)
    print(f"Editor initialized in {time.time() - start_init:.2f}s")
    print_gpu_memory()

    # Run Edit
    print("Starting Edit...")
    start_edit = time.time()
    
    # We use sequential edit for lifelong scenario simulation
    if args.batch_sft:
        # Batch SFT: Pass all requests at once
        edited_model, weights_copy = editor.apply_algo(
            editor.model,
            editor.tok,
            requests,
            editor.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False
        )
        
        # Manual Evaluation
        metrics = []
        eval_requests = requests[:100] if len(requests) > 100 else requests
        print(f"Evaluating on {len(eval_requests)} samples...")
        from tqdm import tqdm
        from easyeditor import compute_edit_quality
        
        for i, request in enumerate(tqdm(eval_requests)):
             res = compute_edit_quality(
                edited_model,
                editor.model_name,
                editor.hparams,
                editor.tok,
                request,
                editor.hparams.device,
                eval_metric=eval_metric
             )
             metrics.append({'post': res})
    else:
        metrics, edited_model, _ = editor.edit_requests(
            requests=requests,
            sequential_edit=True,
            verbose=True,
            eval_metric=eval_metric
        )
    
    end_edit = time.time()
    total_time = end_edit - start_edit
    avg_time = total_time / len(data)
    
    print(f"Editing completed in {total_time:.2f}s")
    print(f"Average time per edit: {avg_time:.2f}s")
    print_gpu_memory()

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_batch" if args.batch_sft else ""
    output_file = os.path.join(args.output_dir, f'{args.method}_{args.data_type}{suffix}_results.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save cost metrics
    cost_file = os.path.join(args.output_dir, f'{args.method}_{args.data_type}{suffix}_cost.json')
    cost_metrics = {
        "total_time": total_time,
        "avg_time_per_edit": avg_time,
        "max_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
    }
    with open(cost_file, 'w') as f:
        json.dump(cost_metrics, f, indent=4)

    print(f"Results saved to {output_file}")
    print(f"Cost metrics saved to {cost_file}")
