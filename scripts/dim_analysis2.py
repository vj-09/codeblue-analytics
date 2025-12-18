#!/usr/bin/env python3
"""Analyze eval results by dimension with question counts."""
import sys
sys.path.insert(0, 'environments/codeblue-analytics')

import json
from pathlib import Path
from collections import defaultdict

# Load questions with metadata
def load_questions_with_meta():
    questions = {}
    for dataset in ['bank', 'road']:
        path = Path(f'environments/codeblue-analytics/data/questions/{dataset}/v1_validated.jsonl')
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        questions[q['id']] = q
    return questions

EVAL_RESULTS = {
    "claude-sonnet-4-5-20250929": {
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": True,
        "bank_v1_009": True, "bank_v1_010": True, "bank_v1_011": True, "bank_v1_012": True,
        "bank_v1_013": True, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": False,
        "bank_v1_017": False, "bank_v1_019": False,
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
    "deepseek/deepseek-chat": {
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": True, "bank_v1_012": True,
        "bank_v1_013": True, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": False,
        "bank_v1_017": False, "bank_v1_019": False,
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
    "gpt-4o": {
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": False,
        "road_v1_009": False, "road_v1_010": False,
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": False,
        "road_v1_009": False, "road_v1_010": False,
    },
    "x-ai/grok-4-fast": {
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": False, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
}

MODEL_SHORT = {
    "claude-sonnet-4-5-20250929": "Sonnet4.5",
    "deepseek/deepseek-chat": "DeepSeek",
    "gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-70b-instruct": "Llama70B",
    "x-ai/grok-4-fast": "Grok4",
}

def analyze():
    questions = load_questions_with_meta()
    dims = ['insight_type', 'business_relevance', 'specificity', 'actionability']
    
    for dim in dims:
        print(f"\n{'='*80}")
        print(f"DIMENSION: {dim}")
        print('='*80)
        
        dim_values = defaultdict(list)
        for qid, q in questions.items():
            grade = q.get('metadata', {}).get('grade', {})
            val = grade.get(dim, 'unknown')
            dim_values[val].append(qid)
        
        print(f"\n{'Value':<18} {'N':>4}", end="")
        for model in EVAL_RESULTS:
            print(f" {MODEL_SHORT[model]:>9}", end="")
        print(f" {'Avg':>8}")
        print("-" * 80)
        
        for val, qids in sorted(dim_values.items(), key=lambda x: str(x[0])):
            n = len(qids)
            print(f"{str(val):<18} {n:>4}", end="")
            val_avgs = []
            for model, results in EVAL_RESULTS.items():
                correct = sum(1 for qid in qids if results.get(qid, False))
                pct = 100 * correct / n if n > 0 else 0
                val_avgs.append(pct)
                print(f" {pct:>8.0f}%", end="")
            print(f" {sum(val_avgs)/len(val_avgs):>7.0f}%")

if __name__ == '__main__':
    analyze()
