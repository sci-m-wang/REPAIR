import sys
import os
import json
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.append('/workspace/REPAIR')

from easyeditor import WISEHyperParams

def load_data(data_dir, data_type, size=50):
    if data_type == 'ZsRE':
        path = f'{data_dir}/{data_type}/zsre_mend_edit.json'
        with open(path, 'r') as f:
            data = json.load(f)[:size]
        prompts = [d['src'] for d in data]
    elif data_type == 'hallucination':
        path = f'{data_dir}/{data_type}/hallucination-edit.json'
        with open(path, 'r') as f:
            data = json.load(f)[:size]
        prompts = [d['prompt'] for d in data]
    elif data_type == 'temporal':
        path = f'{data_dir}/{data_type}/temporal-edit.json'
        with open(path, 'r') as f:
            data = json.load(f)[:size]
        prompts = [d['prompt'] for d in data]
    return prompts

def extract_features(model, tokenizer, prompts, device):
    features = []
    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the last hidden state of the last token
            last_hidden_state = outputs.hidden_states[-1] # [B, L, H]
            # Get last token index (attention_mask)
            last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
            batch_features = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_idx]
            features.append(batch_features.cpu().numpy())
    return np.vstack(features)

def main():
    # Setup
    hparams_path = 'hparams/WISE/llama-3-8b.yaml'
    data_dir = 'data/wise'
    output_dir = 'rebuttal_experiments_final/similarity'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    print(f"Loading model {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True)
    model.eval()

    # Load Data
    print("Loading data...")
    zsre_prompts = load_data(data_dir, 'ZsRE', size=50)
    hallucination_prompts = load_data(data_dir, 'hallucination', size=50)
    
    # Extract Features
    print("Extracting features...")
    zsre_features = extract_features(model, tokenizer, zsre_prompts, device)
    hallucination_features = extract_features(model, tokenizer, hallucination_prompts, device)
    
    # Create Batches
    # Homogeneous: ZsRE only vs Hallucination only
    # Heterogeneous: Mix
    
    # Compute Similarity
    print("Computing similarity...")
    
    # Intra-class similarity (Homogeneous)
    sim_zsre = cosine_similarity(zsre_features)
    sim_hallucination = cosine_similarity(hallucination_features)
    
    # Inter-class similarity (Heterogeneous)
    sim_cross = cosine_similarity(zsre_features, hallucination_features)
    
    # Flatten and remove self-similarity for intra-class
    mask_zsre = ~np.eye(sim_zsre.shape[0], dtype=bool)
    vals_zsre = sim_zsre[mask_zsre]
    
    mask_hall = ~np.eye(sim_hallucination.shape[0], dtype=bool)
    vals_hall = sim_hallucination[mask_hall]
    
    vals_cross = sim_cross.flatten()
    
    # Plot Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(vals_zsre, bins=50, alpha=0.5, label='ZsRE (Homo)', density=True)
    plt.hist(vals_hall, bins=50, alpha=0.5, label='Hallucination (Homo)', density=True)
    plt.hist(vals_cross, bins=50, alpha=0.5, label='Cross (Hetero)', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Feature Similarity Distribution')
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    plt.close()
    
    # t-SNE
    print("Running t-SNE...")
    all_features = np.vstack([zsre_features, hallucination_features])
    labels = ['ZsRE'] * len(zsre_features) + ['Hallucination'] * len(hallucination_features)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    for label in ['ZsRE', 'Hallucination']:
        mask = [l == label for l in labels]
        plt.scatter(embedded[mask, 0], embedded[mask, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of Features')
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    
    # Save stats
    stats = {
        'mean_sim_zsre': float(np.mean(vals_zsre)),
        'mean_sim_hallucination': float(np.mean(vals_hall)),
        'mean_sim_cross': float(np.mean(vals_cross))
    }
    with open(os.path.join(output_dir, 'similarity_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
