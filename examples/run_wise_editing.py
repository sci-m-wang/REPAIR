import os.path
import sys
import json
import argparse
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('..')
# Add ELDER peft_egg to sys.path
sys.path.insert(0, "/workspace/REPAIR/ELDER/peft_egg/src")

from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)

# Import ELDER components from run_elder.py (assuming it's in root)
try:
    from run_elder import ElderHyperParams, apply_elder_to_model, train_elder
    from easyeditor.evaluate import compute_rewrite_or_rephrase_quality, compute_locality_quality
except ImportError:
    print("Warning: Could not import ELDER components. ELDER method will fail.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=False, type=str,default='WISE')
    parser.add_argument('--hparams_dir', required=False, type=str,default='../hparams/WISE/llama-3-8b.yaml')
    parser.add_argument('--data_dir', required=False, type=str,default='../data/wise')
    parser.add_argument('--data_type', required=False, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination'],default='ZsRE')
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--output_file', default=None, type=str) # Added output_file arg
    parser.add_argument('--ds_size', default=10, type=int)
    parser.add_argument('--sequential_edit', action="store_true",default=True)


    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ELDER':
        editing_hparams = ElderHyperParams
    else:
        raise NotImplementedError

    K =args.ds_size


    if args.data_type == 'ZsRE':
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
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'temporal':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['ood_rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    
    # Override device if needed (e.g. if using CUDA_VISIBLE_DEVICES)
    # hparams.device = 0 

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
            args.output_dir,
            f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
            )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl'
    }

    if args.editing_method == 'ELDER':
        # ELDER Custom Execution Loop
        print(f"Loading model {hparams.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(hparams.model_name, device_map=hparams.device, trust_remote_code=True)
        
        print("Applying ELDER...")
        model = apply_elder_to_model(model, hparams)
        model.to(hparams.device) # Ensure new layers are on GPU
        
        metrics = []
        for i in range(len(prompts)):
            print(f"Editing sample {i+1}/{len(prompts)}: {prompts[i]}")
            
            # Construct request object for ELDER
            req = {
                'prompt': prompts[i],
                'target_new': target_new[i],
                'locality': {
                    'neighborhood': {
                        'prompt': locality_inputs['neighborhood']['prompt'][i],
                        'ground_truth': locality_inputs['neighborhood']['ground_truth'][i]
                    }
                }
            }
            
            sample_metrics = {
                "case_id": i,
                "pre": {},
                "post": {}
            }
            
            # Pre-edit Eval
            sample_metrics["pre"]["rewrite"] = compute_rewrite_or_rephrase_quality(
                model=model, model_name=hparams.model_name, hparams=hparams, tok=tokenizer,
                prompt=req['prompt'], target_new=req['target_new'], device=hparams.device
            )
            sample_metrics["pre"]["locality"] = compute_locality_quality(
                model=model, model_name=hparams.model_name, hparams=hparams, tok=tokenizer,
                locality_key='neighborhood',
                prompt=req['locality']['neighborhood']['prompt'],
                locality_ground_truth=req['locality']['neighborhood']['ground_truth'],
                device=hparams.device
            )
            
            # Train
            model, losses = train_elder(model, tokenizer, req, hparams)
            print(f"Final Loss: {losses[-1]}")
            
            # Post-edit Eval
            model.eval()
            for module in model.modules():
                if hasattr(module, 'editing'): module.editing = False
            
            sample_metrics["post"]["rewrite"] = compute_rewrite_or_rephrase_quality(
                model=model, model_name=hparams.model_name, hparams=hparams, tok=tokenizer,
                prompt=req['prompt'], target_new=req['target_new'], device=hparams.device
            )
            sample_metrics["post"]["locality"] = compute_locality_quality(
                model=model, model_name=hparams.model_name, hparams=hparams, tok=tokenizer,
                locality_key='neighborhood',
                prompt=req['locality']['neighborhood']['prompt'],
                locality_ground_truth=req['locality']['neighborhood']['ground_truth'],
                device=hparams.device
            )
            
            metrics.append(sample_metrics)

    else:
        # Standard Editor Execution
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
            eval_metric=eval_metric[args.data_type]
        )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)
