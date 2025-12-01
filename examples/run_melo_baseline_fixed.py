#!/usr/bin/env python3
"""
运行MELO baseline实验 - 修正版
直接使用MELO的apply_melo_to_model函数
"""

import os
import sys
import json
import argparse
import torch
sys.path.append('..')

from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.models.melo import MELOHyperParams, apply_melo_to_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=False, type=str, default='../data/wise')
    parser.add_argument('--data_type', required=False, type=str,
                        choices=['ZsRE', 'Temporal', 'Hallucination'], default='ZsRE')
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=100, type=int)
    
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
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tok = AutoTokenizer.from_pretrained(hparams.model_name)
    tok.pad_token = tok.eos_token
    
    # Prepare results
    results = []
    K = args.ds_size
    
    print(f"Running MELO on {K} samples...")
    
    # Process each edit sequentially
    for i in range(K):
        print(f"\nProcessing edit {i+1}/{K}...")
        
        # Format request for MELO
        # MELO expects: {"text": prompt, "labels": target}
        request = {
            "text": data[i]['src'],
            "labels": data[i]['alt']
        }
        
        # Apply MELO edit
        model, _ = apply_melo_to_model(
            model=model,
            tok=tok,
            requests=[request],  # MELO expects list of requests
            hparams=hparams,
            keep_original_weight=False
        )
        
        # Test the edit
        with torch.no_grad():
            # Test rewrite
            inputs = tok(data[i]['src'], return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=20)
            rewrite_output = tok.decode(outputs[0], skip_special_tokens=True)
            
            # Test rephrase
            inputs_rephrase = tok(data[i]['rephrase'], return_tensors='pt').to(model.device)
            outputs_rephrase = model.generate(**inputs_rephrase, max_new_tokens=20)
            rephrase_output = tok.decode(outputs_rephrase[0], skip_special_tokens=True)
            
            # Test locality
            inputs_loc = tok(data[i]['loc'], return_tensors='pt').to(model.device)
            outputs_loc = model.generate(**inputs_loc, max_new_tokens=20)
            locality_output = tok.decode(outputs_loc[0], skip_special_tokens=True)
        
        # Store results
        result = {
            'case_id': i,
            'requested_rewrite': {
                'prompt': data[i]['src'],
                'target_new': data[i]['alt'],
                'subject': data[i]['subject']
            },
            'post': {
                'rewrite_output': rewrite_output,
                'rephrase_output': rephrase_output,
                'locality_output': locality_output
            }
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{K} edits")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'MELO_{args.data_type}_N{args.ds_size}.json'
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"✓ Completed {K} edits successfully")
