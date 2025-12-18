#!/usr/bin/env python3
"""Analyze pass/fail by difficulty level."""
import json

# Load the question file to get levels
questions = []
with open('environments/codeblue-analytics/data/questions/bank/v2_hard.jsonl') as f:
    for line in f:
        questions.append(json.loads(line))

# Create level mapping (convert L3 -> 3, L4 -> 4, L5 -> 5)
level_map = {q['id']: int(q['level'][1]) for q in questions}

# Load evaluation results
results = []
with open('outputs/evals/bank_v2_hard_deepseek.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Analyze by level
level_stats = {3: {'pass': 0, 'fail': 0}, 4: {'pass': 0, 'fail': 0}, 5: {'pass': 0, 'fail': 0}}

for r in results:
    qid = r['question_id']
    level = level_map.get(qid, 0)
    
    if level in level_stats:
        if r['correct']:
            level_stats[level]['pass'] += 1
        else:
            level_stats[level]['fail'] += 1

# Print results
print('\n📊 PASS/FAIL BREAKDOWN BY DIFFICULTY LEVEL')
print('=' * 70)
print(f"{'Level':<8} {'Total':<8} {'Pass':<8} {'Fail':<8} {'Accuracy':<12} {'Visual'}")
print('-' * 70)

for level in [3, 4, 5]:
    stats = level_stats[level]
    total = stats['pass'] + stats['fail']
    if total > 0:
        accuracy = 100 * stats['pass'] / total
        bar_pass = '█' * stats['pass']
        bar_fail = '░' * stats['fail']
        print(f"L{level:<7} {total:<8} {stats['pass']:<8} {stats['fail']:<8} {accuracy:>5.1f}%      {bar_pass}{bar_fail}")

# Overall
total_all = sum(s['pass'] + s['fail'] for s in level_stats.values())
pass_all = sum(s['pass'] for s in level_stats.values())
fail_all = sum(s['fail'] for s in level_stats.values())
acc_all = 100 * pass_all / total_all

print('-' * 70)
print(f"{'Total':<8} {total_all:<8} {pass_all:<8} {fail_all:<8} {acc_all:>5.1f}%")

# Detailed breakdown
print('\n📈 DETAILED BREAKDOWN')
print('=' * 70)
for level in [3, 4, 5]:
    stats = level_stats[level]
    total = stats['pass'] + stats['fail']
    if total > 0:
        accuracy = 100 * stats['pass'] / total
        print(f"\nLevel {level} (Total: {total} questions):")
        print(f"  ✅ Pass: {stats['pass']:2} ({100*stats['pass']/total:5.1f}%)")
        print(f"  ❌ Fail: {stats['fail']:2} ({100*stats['fail']/total:5.1f}%)")
        print(f"  Difficulty: {'Easy' if level == 3 else 'Medium' if level == 4 else 'Hard'}")
