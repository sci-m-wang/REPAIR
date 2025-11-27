import json
import glob
import os
import numpy as np

def load_results(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(results):
    if not results:
        return None
    
    rewrite_accs = []
    locality_accs = []
    
    for res in results:
        # Rewrite Accuracy
        # Check if 'post' has 'rewrite_acc' or 'rewrite_output'
        if 'post' in res:
            post = res['post']
            if 'rewrite_acc' in post:
                rewrite_accs.append(post['rewrite_acc'][0])
            elif 'rewrite' in post: # ELDER structure
                 # Need to parse this if it's not a simple acc
                 pass
            
            # Locality Accuracy
            if 'locality_acc' in post:
                locality_accs.append(post['locality_acc'][0])
            elif 'locality' in post:
                 if 'neighborhood_acc' in post['locality']:
                     locality_accs.append(post['locality']['neighborhood_acc'][0])

    metrics = {
        "Rewrite Acc": np.mean(rewrite_accs) if rewrite_accs else 0.0,
        "Locality Acc": np.mean(locality_accs) if locality_accs else 0.0,
        "Count": len(results)
    }
    return metrics

datasets = ["custom", "mental_health", "medical"]
methods = ["FT", "REPAIR", "WISE", "GRACE"]

print(f"{'Dataset':<15} {'Method':<10} {'Rewrite Acc':<15} {'Locality Acc':<15} {'Count':<5}")
print("-" * 60)

for dataset in datasets:
    for method in methods:
        # Construct filename
        if dataset == "custom":
            if method == "FT":
                filename = "outputs_comparison/FT_results.json"
            else:
                filename = f"outputs_comparison/{method}_custom_results.json"
        else:
            filename = f"outputs_comparison/{method}_{dataset}_results.json"
            
        results = load_results(filename)
        metrics = calculate_metrics(results)
        
        if metrics:
            print(f"{dataset:<15} {method:<10} {metrics['Rewrite Acc']:<15.4f} {metrics['Locality Acc']:<15.4f} {metrics['Count']:<5}")
        else:
            print(f"{dataset:<15} {method:<10} {'N/A':<15} {'N/A':<15} {'0':<5}")
