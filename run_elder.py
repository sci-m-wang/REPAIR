import sys
import os
import torch
import json
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy

# Add ELDER peft_egg to sys.path
sys.path.insert(0, "/workspace/REPAIR/ELDER/peft_egg/src")

from peft.tuners.elder import ELDERConfig, ELDERModel
from easyeditor.util.hparams import HyperParams
from easyeditor.evaluate import compute_rewrite_or_rephrase_quality, compute_locality_quality

from dataclasses import dataclass

@dataclass
class ElderHyperParams(HyperParams):
    alg_name: str
    model_name: str
    device: int
    r: int
    lora_alpha: int
    lora_dropout: float
    num_experts: int
    is_redundant_experts: bool
    grace_layer: str
    grace_config: dict
    lr: float
    n_iter: int
    max_grad_norm: float
    max_length: int = 2048

def apply_elder_to_model(model, hparams):
    # Parse grace_layer index
    try:
        grace_layer_idx = int(hparams.grace_layer.split('.')[2])
        # Transform layers AFTER the grace layer
        layers_to_transform = list(range(grace_layer_idx + 1, 28)) # Qwen2.5-7B has 28 layers
    except:
        print("Warning: Could not parse grace_layer index. Using default layers_to_transform=None (might fail).")
        layers_to_transform = None

    config = ELDERConfig(
        r=hparams.r,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=["down_proj"], # Qwen MLP down_proj
        grace_layer=hparams.grace_layer,
        grace_config=hparams.grace_config,
        num_experts=hparams.num_experts,
        is_redundant_experts=hparams.is_redundant_experts,
        layers_to_transform=layers_to_transform
    )
    model = ELDERModel(model, {"default": config}, "default")
    return model

def train_elder(model, tokenizer, request, hparams):
    model.train()
    # Set adapter to trainable
    # model.enable_adapter_layers() # Should be enabled by default
    
    # Prepare inputs
    prompt = request['prompt']
    target = request['target_new']
    
    # Tokenize
    inputs = tokenizer(prompt + target, return_tensors="pt").to(model.device)
    target_ids = inputs['input_ids'].clone()
    
    # Mask prompt in target
    prompt_len = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
    target_ids[:, :prompt_len] = -100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    
    losses = []
    
    # Find the GraceLayer to set batch_iter
    grace_module = None
    for module in model.modules():
        if hasattr(module, 'VecDB'): # Identify GraceLayer
            grace_module = module
            break
            
    if grace_module is None:
        print("Warning: GraceLayer not found!")

    # Set editing mode and labels
    for module in model.modules():
        if hasattr(module, 'editing'):
            module.editing = True
        if hasattr(module, 'edit_label'):
            module.edit_label = target_ids # ElderGraceLinear needs this

    for i in range(hparams.n_iter):
        if grace_module:
            grace_module.batch_iter = i
            
        outputs = model(**inputs, labels=target_ids)
        loss = outputs.loss
        
        # Add gate loss if available
        gate_loss = 0
        for module in model.modules():
            if hasattr(module, 'gate_loss') and module.gate_loss is not None:
                gate_loss += module.gate_loss
        
        total_loss = loss + gate_loss
        
        total_loss.backward()
        
        if hparams.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.max_grad_norm)
            
        optimizer.step()
        optimizer.zero_grad()
        losses.append(total_loss.item())
        
    return model, losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ds_size', type=int, default=None)
    args = parser.parse_args()
    
    import yaml
    with open(args.hparams, 'r') as f:
        hparams_dict = yaml.safe_load(f)
    hparams = ElderHyperParams(**hparams_dict)
    
    # Load Model
    print(f"Loading model {hparams.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(hparams.model_name, device_map=hparams.device, trust_remote_code=True)
    
    # Apply ELDER
    print("Applying ELDER...")
    model = apply_elder_to_model(model, hparams)
    # model.print_trainable_parameters()
    
    # Load Data
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    if args.ds_size:
        data = data[:args.ds_size]
        
    results = []
    
    for i, req in enumerate(data):
        print(f"Editing sample {i+1}/{len(data)}: {req['prompt']}")
        
        # Pre-edit evaluation
        metrics = {
            "case_id": i,
            "pre": {},
            "post": {}
        }
        
        # Rewrite Eval
        metrics["pre"]["rewrite"] = compute_rewrite_or_rephrase_quality(
            model=model,
            model_name=hparams.model_name,
            hparams=hparams,
            tok=tokenizer,
            prompt=req['prompt'],
            target_new=req['target_new'],
            device=hparams.device
        )
        
        # Locality Eval
        if 'locality' in req:
            metrics["pre"]["locality"] = compute_locality_quality(
                model=model,
                model_name=hparams.model_name,
                hparams=hparams,
                tok=tokenizer,
                locality_key='neighborhood',
                prompt=req['locality']['neighborhood']['prompt'],
                locality_ground_truth=req['locality']['neighborhood']['ground_truth'],
                device=hparams.device
            )
        
        # Train
        model, losses = train_elder(model, tokenizer, req, hparams)
        print(f"Final Loss: {losses[-1]}")
        
        # Post-edit evaluation
        model.eval()
        for module in model.modules():
            if hasattr(module, 'editing'):
                module.editing = False
        
        metrics["post"]["rewrite"] = compute_rewrite_or_rephrase_quality(
            model=model,
            model_name=hparams.model_name,
            hparams=hparams,
            tok=tokenizer,
            prompt=req['prompt'],
            target_new=req['target_new'],
            device=hparams.device
        )
        
        if 'locality' in req:
            metrics["post"]["locality"] = compute_locality_quality(
                model=model,
                model_name=hparams.model_name,
                hparams=hparams,
                tok=tokenizer,
                locality_key='neighborhood',
                prompt=req['locality']['neighborhood']['prompt'],
                locality_ground_truth=req['locality']['neighborhood']['ground_truth'],
                device=hparams.device
            )
        
        results.append(metrics)
        
    # Save results
    output_file = f"outputs_comparison/ELDER_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
