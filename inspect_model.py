
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/workspace/models/Qwen/Qwen2.5-7B-Instruct"
print(f"Loading model from {model_path}")
try:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print("Model loaded")
    print("Named parameters:")
    for name, param in model.named_parameters():
        print(name)
        if "layers.0." in name: # Print only first layer details to save space
             pass
        else:
             if "layers" in name:
                 continue # Skip other layers
    
    # Print a few specific ones to be sure
    print("\nSpecific modules:")
    print(model)
except Exception as e:
    print(f"Error: {e}")
