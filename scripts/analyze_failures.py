#!/usr/bin/env python3
"""
Analyze failure patterns from evaluation results.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

def analyze_failures(jsonl_path):
    """Analyze failures from JSONL results file."""
    
    # Read results
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    
    failures = [r for r in results if not r['correct']]
    
    print("=" * 80)
    print(f"FAILURE ANALYSIS: {jsonl_path.name}")
    print("=" * 80)
    print(f"\nTotal: {len(results)} questions")
    print(f"Passed: {len(results) - len(failures)} ({100*(len(results)-len(failures))/len(results):.1f}%)")
    print(f"Failed: {len(failures)} ({100*len(failures)/len(results):.1f}%)")
    
    # Categorize failures
    categories = {
        'execution_error': [],
        'format_mismatch_df': [],
        'format_mismatch_scalar': [],
        'unit_mismatch': [],
        'sorting_order': [],
        'precision_rounding': [],
        'logic_error': []
    }
    
    for f in failures:
        qid = f['question_id']
        question = f['question']
        result = f['result']
        expected = f['expected']
        error = f.get('error', '')
        
        # Execution error
        if not f['exec_success']:
            categories['execution_error'].append({
                'qid': qid,
                'question': question,
                'error': error
            })
            continue
        
        # Format mismatch - DataFrame expected but not returned
        if 'DataFrame' in str(expected) and 'DataFrame' not in str(result):
            categories['format_mismatch_df'].append({
                'qid': qid,
                'question': question,
                'expected_type': 'DataFrame',
                'got_type': type(result).__name__
            })
            continue
        
        # Format mismatch - scalar expected but got tuple/other
        if isinstance(expected, (int, float)) and isinstance(result, str) and '(' in str(result):
            categories['format_mismatch_scalar'].append({
                'qid': qid,
                'question': question,
                'expected': expected,
                'got': result
            })
            continue
        
        # Unit mismatch (percentage: 0.5 vs 50)
        if isinstance(expected, (int, float)) and isinstance(result, str):
            try:
                result_val = float(result)
                if abs(result_val * 100 - expected) < 1.0 or abs(result_val - expected * 100) < 1.0:
                    categories['unit_mismatch'].append({
                        'qid': qid,
                        'question': question,
                        'expected': expected,
                        'got': result
                    })
                    continue
            except:
                pass
        
        # Sorting order mismatch
        if isinstance(expected, list) and isinstance(result, str) and '[' in result:
            categories['sorting_order'].append({
                'qid': qid,
                'question': question,
                'note': 'List order mismatch'
            })
            continue
        
        # Precision/rounding
        try:
            if result is not None and expected is not None:
                result_num = float(str(result).replace('%', '').replace(',', ''))
                expected_num = float(str(expected).replace('%', '').replace(',', ''))
                if abs(result_num - expected_num) < 1.0:
                    categories['precision_rounding'].append({
                        'qid': qid,
                        'question': question,
                        'expected': expected,
                        'got': result,
                        'diff': abs(result_num - expected_num)
                    })
                    continue
        except:
            pass
        
        # Default to logic error
        categories['logic_error'].append({
            'qid': qid,
            'question': question,
            'expected': str(expected)[:100],
            'got': str(result)[:100]
        })
    
    # Print categorized failures
    print("\n" + "=" * 80)
    print("FAILURE CATEGORIES")
    print("=" * 80)
    
    for category, items in categories.items():
        if not items:
            continue
        
        print(f"\n### {category.upper().replace('_', ' ')} ({len(items)} questions)")
        print("-" * 80)
        
        for item in items[:5]:  # Show first 5
            print(f"\n{item['qid']}:")
            print(f"  Q: {item['question'][:80]}...")
            if 'error' in item:
                print(f"  Error: {item['error']}")
            if 'expected' in item and 'got' in item:
                print(f"  Expected: {item['expected']}")
                print(f"  Got: {item['got']}")
            if 'note' in item:
                print(f"  Note: {item['note']}")
        
        if len(items) > 5:
            print(f"\n  ... and {len(items) - 5} more")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for category, items in categories.items():
        if items:
            pct = 100 * len(items) / len(failures)
            print(f"  {category:25} {len(items):3} ({pct:5.1f}% of failures)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_failures.py <results.jsonl>")
        sys.exit(1)
    
    analyze_failures(Path(sys.argv[1]))
