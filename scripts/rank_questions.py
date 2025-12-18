#!/usr/bin/env python3
"""Rank questions by quality for trimming."""
import json
from pathlib import Path

def score_question(q):
    """Score a question (higher = better quality, keep)."""
    grade = q.get('metadata', {}).get('grade', {})
    score = 0
    
    # business_relevance: 3 > 2 > 1
    br = grade.get('business_relevance', 2)
    score += br * 10
    
    # insight_type: diagnostic > comparative > descriptive
    it = grade.get('insight_type', 'descriptive')
    it_scores = {'diagnostic': 30, 'comparative': 20, 'descriptive': 10}
    score += it_scores.get(it, 10)
    
    # specificity: context-specific > domain-aware > generic
    sp = grade.get('specificity', 'generic')
    sp_scores = {'context-specific': 15, 'domain-aware': 10, 'generic': 5}
    score += sp_scores.get(sp, 5)
    
    # actionability: yes > partial > no
    ac = grade.get('actionability', 'no')
    ac_scores = {'yes': 15, 'partial': 10, 'no': 5}
    score += ac_scores.get(ac, 5)
    
    # overall_value: high > medium > low
    ov = grade.get('overall_value', 'medium')
    ov_scores = {'high': 20, 'medium': 10, 'low': 0}
    score += ov_scores.get(ov, 10)
    
    return score

all_questions = []
for dataset in ['bank', 'road', 'shopping']:
    path = Path(f'environments/codeblue-analytics/data/questions/{dataset}/v1_validated.jsonl')
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    q['_score'] = score_question(q)
                    q['_dataset'] = dataset
                    all_questions.append(q)

# Sort by score (lowest first = cut candidates)
all_questions.sort(key=lambda x: x['_score'])

print(f"Total questions: {len(all_questions)}")
print(f"Need to cut: {len(all_questions) - 50}")
print()
print("=== LOWEST SCORED (CUT CANDIDATES) ===")
print(f"{'ID':<20} {'Dataset':<10} {'Score':<6} {'Question':<60}")
print("-" * 100)
for q in all_questions[:20]:
    qtext = q['question'][:57] + "..." if len(q['question']) > 60 else q['question']
    print(f"{q['id']:<20} {q['_dataset']:<10} {q['_score']:<6} {qtext}")

print()
print("=== DATASET BREAKDOWN ===")
for ds in ['bank', 'road', 'shopping']:
    ds_qs = [q for q in all_questions if q['_dataset'] == ds]
    print(f"{ds}: {len(ds_qs)} questions, avg score: {sum(q['_score'] for q in ds_qs)/len(ds_qs):.1f}")
