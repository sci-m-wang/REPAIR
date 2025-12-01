import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/workspace/REPAIR')

from easyeditor import WISEHyperParams, BaseEditor
import lm_eval
from lm_eval.models.huggingface import HFLM

def evaluate_reasoning(model, tokenizer, tasks=['gsm8k', 'mmlu'], limit=100):
    print(f"Evaluating on {tasks} with limit={limit}...")
    
    # Wrap model for lm_eval
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit
    )
    
    # Extract scores
    scores = {}
    for task in tasks:
        # lm_eval structure varies by version, trying to be robust
        if 'results' in results:
            if task in results['results']:
                # Usually 'acc,none' or just 'acc'
                res = results['results'][task]
                score = res.get('acc,none', res.get('acc', 0.0))
                scores[task] = score
            else:
                # Handle MMLU subtasks if necessary, or aggregated
                # For simplicity, just taking what's there
                pass
    
    # If mmlu is aggregated
    if 'mmlu' in tasks and 'mmlu' not in scores:
         # Aggregate mmlu subtasks if present
         mmlu_scores = [v.get('acc,none', v.get('acc', 0.0)) for k, v in results['results'].items() if 'mmlu' in k]
         if mmlu_scores:
             scores['mmlu'] = np.mean(mmlu_scores)

    return scores, results

def main():
    # Setup
    hparams_path = 'hparams/WISE/llama-3-8b.yaml'
    data_dir = 'data/wise'
    data_type = 'ZsRE'
    ds_size = 100
    output_dir = 'rebuttal_experiments_final/reasoning'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = "/root/workspace/share/LLM-Research/Meta-Llama-3-8B-Instruct"

    # Load Hparams
    hparams = WISEHyperParams.from_hparams(hparams_path)
    hparams.model_name = model_path
    hparams.device = 0 # Force GPU 0 (relative to visible devices)

    # Initialize Editor (loads model)
    print("Initializing Editor and Model...")
    editor = BaseEditor.from_hparams(hparams)
    model = editor.model
    tokenizer = editor.tok

    # 1. Pre-edit Evaluation
    print("\n[Phase 1] Pre-edit Evaluation")
    pre_scores, pre_results = evaluate_reasoning(model, tokenizer)
    print(f"Pre-edit Scores: {pre_scores}")

    # 2. Perform Editing
    print("\n[Phase 2] Performing Editing (ZsRE N=100)")
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

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=locality_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=True,
        eval_metric='token em'
    )
    
    # 3. Post-edit Evaluation
    print("\n[Phase 3] Post-edit Evaluation")
    # Note: edited_model is the modified model (WISE wrapper)
    # Unwrap if necessary
    if hasattr(edited_model, 'model'):
        model_to_eval = edited_model.model
    else:
        model_to_eval = edited_model
        
    post_scores, post_results = evaluate_reasoning(model_to_eval, tokenizer)
    print(f"Post-edit Scores: {post_scores}")

    # 4. Calculate Locality Retention
    retention = {}
    for task in pre_scores:
        if pre_scores[task] > 0:
            retention[task] = post_scores.get(task, 0.0) / pre_scores[task]
        else:
            retention[task] = 0.0
    
    print(f"\nLocality Retention: {retention}")

    # Save Results
    final_results = {
        'pre_scores': pre_scores,
        'post_scores': post_scores,
        'retention': retention,
        'edit_metrics': metrics
    }
    
    output_file = os.path.join(output_dir, 'reasoning_locality_results.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
