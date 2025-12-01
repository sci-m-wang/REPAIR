#!/usr/bin/env python3
"""
转换REPAIR数据格式到MELO/ELDER格式
"""

import json
import jsonlines

def convert_zsre_to_melo_format():
    """
    REPAIR格式:
    {
        "src": "question",
        "alt": "answer", 
        "subject": "entity",
        "rephrase": "rephrase question",
        "loc": "locality question",
        "loc_ans": "locality answer"
    }
    
    MELO格式 (zsre_dev.jsonl):
    {
        "input": "question",
        "output": [{"answer": "answer"}],
        "prediction": "old_answer",
        "alternatives": ["alt1", "alt2"],
        "filtered_rephrases": ["rephrase1", "rephrase2", ...]
    }
    """
    
    # 读取REPAIR数据
    with open('/root/REPAIR/data/wise/ZsRE/zsre_mend_edit.json', 'r') as f:
        repair_data = json.load(f)
    
    # 转换为MELO格式
    melo_data = []
    for item in repair_data:
        melo_item = {
            "input": item['src'],
            "output": [{"answer": item['alt']}],
            "prediction": "",  # 原始答案，REPAIR数据中没有
            "alternatives": [item['alt']],  # 使用alt作为答案
            "filtered_rephrases": [item['rephrase']] * 10  # MELO需要至少10个rephrase
        }
        melo_data.append(melo_item)
    
    # 保存为jsonl格式
    output_path = '/root/REPAIR/external_baselines/MELO/melo/data/zsre_dev.jsonl'
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(melo_data)
    
    print(f"✓ Converted {len(melo_data)} zsRE samples to MELO format")
    print(f"  Saved to: {output_path}")
    return len(melo_data)

def create_nq_train_from_zsre():
    """
    创建NQ训练数据（使用zsRE数据的一部分）
    
    NQ格式:
    {
        "questions": ["q1", "q2", ...],
        "answers": ["a1", "a2", ...]
    }
    """
    
    # 读取REPAIR数据
    with open('/root/REPAIR/data/wise/ZsRE/zsre_mend_edit.json', 'r') as f:
        repair_data = json.load(f)
    
    # 使用前1000个样本作为"训练"数据
    questions = [item['src'] for item in repair_data[:1000]]
    answers = [item['alt'] for item in repair_data[:1000]]
    
    nq_data = {
        "questions": questions,
        "answers": answers
    }
    
    # 保存
    output_path = '/root/REPAIR/external_baselines/MELO/melo/data/nq_train.json'
    with open(output_path, 'w') as f:
        json.dump(nq_data, f, indent=2)
    
    print(f"✓ Created NQ train data with {len(questions)} samples")
    print(f"  Saved to: {output_path}")
    return len(questions)

if __name__ == "__main__":
    print("Converting REPAIR data to MELO/ELDER format...")
    print("="*60)
    
    # 转换zsRE数据
    n_zsre = convert_zsre_to_melo_format()
    
    # 创建NQ训练数据
    n_nq = create_nq_train_from_zsre()
    
    print("="*60)
    print("Conversion complete!")
    print(f"  zsRE samples: {n_zsre}")
    print(f"  NQ train samples: {n_nq}")
