
import csv
import json
import os

csv_path = "/workspace/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
output_path = "data/wise/custom_qwen_test.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 100: # Limit to 100 examples
            break
        data.append({
            "prompt": row['instruction'],
            "target_new": row['response'],
            "ground_truth": row['response'],
            "loc_prompt": "What is the capital of France?", # Dummy
            "loc_ans": "Paris" # Dummy
        })

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} examples to {output_path}")

# Mental Health Dataset (Full)
mh_path = "/workspace/datasets/Amod/mental_health_counseling_conversations/combined_dataset.json"
mh_output_path = "data/wise/mental_health_full.json"
mh_data = []
with open(mh_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # if i >= 10: break # Use full dataset
        row = json.loads(line)
        mh_data.append({
            "prompt": row['Context'],
            "target_new": row['Response'],
            "ground_truth": row['Response'],
            "loc_prompt": "What is the capital of France?",
            "loc_ans": "Paris"
        })

with open(mh_output_path, 'w', encoding='utf-8') as f:
    json.dump(mh_data, f, indent=2)
print(f"Converted {len(mh_data)} examples to {mh_output_path}")

# Medical Dataset (Full)
med_path = "/workspace/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/medical_o1_sft.json"
med_output_path = "data/wise/medical_full.json"
med_data = []
with open(med_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    for i, row in enumerate(raw_data):
        # if i >= 10: break # Use full dataset
        med_data.append({
            "prompt": row['Question'],
            "target_new": row['Response'],
            "ground_truth": row['Response'],
            "loc_prompt": "What is the capital of France?",
            "loc_ans": "Paris"
        })

with open(med_output_path, 'w', encoding='utf-8') as f:
    json.dump(med_data, f, indent=2)
print(f"Converted {len(med_data)} examples to {med_output_path}")
