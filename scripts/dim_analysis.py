#!/usr/bin/env python3
"""Analyze eval results by dimension."""
import sys
sys.path.insert(0, 'environments/codeblue-analytics')

import json
import asyncio
from pathlib import Path
from collections import defaultdict
from codeblue_analytics import load_environment, load_dataframes, extract_code, compare_outputs

# Load questions with metadata
def load_questions_with_meta():
    """Load all questions with their grading metadata."""
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

# Eval results from previous runs (hardcoded since we didn't save them)
# Format: {model: {question_id: correct}}
EVAL_RESULTS = {
    "claude-sonnet-4-5-20250929": {
        # Bank: 72% (13/18)
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": True,
        "bank_v1_009": True, "bank_v1_010": True, "bank_v1_011": True, "bank_v1_012": True,
        "bank_v1_013": True, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": False,
        "bank_v1_017": False, "bank_v1_019": False,
        # Road: 80% (8/10)
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
    "deepseek/deepseek-chat": {
        # Bank: 61% (11/18)
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": True, "bank_v1_012": True,
        "bank_v1_013": True, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": False,
        "bank_v1_017": False, "bank_v1_019": False,
        # Road: 80% (8/10)
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
    "gpt-4o": {
        # Bank: 56% (10/18)
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        # Road: 70% (7/10)
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": False,
        "road_v1_009": False, "road_v1_010": False,
    },
    "meta-llama/llama-3.1-70b-instruct": {
        # Bank: 56% (10/18)
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": True, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        # Road: 70% (7/10)
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": False,
        "road_v1_009": False, "road_v1_010": False,
    },
    "x-ai/grok-4-fast": {
        # Bank: 50% (9/18)
        "bank_v1_001": True, "bank_v1_002": True, "bank_v1_003": True, "bank_v1_004": True,
        "bank_v1_005": True, "bank_v1_006": True, "bank_v1_007": True, "bank_v1_008": False,
        "bank_v1_009": False, "bank_v1_010": False, "bank_v1_011": False, "bank_v1_012": True,
        "bank_v1_013": False, "bank_v1_014": False, "bank_v1_015": False, "bank_v1_016": True,
        "bank_v1_017": False, "bank_v1_019": False,
        # Road: 80% (8/10)
        "road_v1_001": True, "road_v1_002": True, "road_v1_003": True, "road_v1_004": True,
        "road_v1_005": True, "road_v1_006": True, "road_v1_007": True, "road_v1_008": True,
        "road_v1_009": False, "road_v1_010": False,
    },
}

MODEL_SHORT = {
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "deepseek/deepseek-chat": "DeepSeek",
    "gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-70b-instruct": "Llama 70B",
    "x-ai/grok-4-fast": "Grok 4",
}

def analyze():
    questions = load_questions_with_meta()
    
    # Analyze by dimension
    dims = ['insight_type', 'business_relevance', 'specificity', 'actionability']
    
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"DIMENSION: {dim}")
        print('='*60)
        
        # Group questions by dimension value
        dim_values = defaultdict(list)
        for qid, q in questions.items():
            grade = q.get('metadata', {}).get('grade', {})
            val = grade.get(dim, 'unknown')
            dim_values[val].append(qid)
        
        # For each dimension value, calculate model accuracy
        print(f"\n{'Value':<20}", end="")
        for model in EVAL_RESULTS:
            print(f"{MODEL_SHORT[model]:>12}", end="")
        print(f"{'Avg':>10}")
        print("-" * (20 + 12*len(EVAL_RESULTS) + 10))
        
        for val, qids in sorted(dim_values.items(), key=lambda x: str(x[0])):
            print(f"{str(val):<20}", end="")
            val_avgs = []
            for model, results in EVAL_RESULTS.items():
                correct = sum(1 for qid in qids if results.get(qid, False))
                total = len(qids)
                pct = 100 * correct / total if total > 0 else 0
                val_avgs.append(pct)
                print(f"{pct:>11.0f}%", end="")
            print(f"{sum(val_avgs)/len(val_avgs):>9.0f}%")

if __name__ == '__main__':
    analyze()
