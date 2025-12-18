import json
import sys
from pathlib import Path
from collections import Counter

def summarize_results(results_dir):
    results_path = Path(results_dir)
    json_files = sorted(list(results_path.glob("*.json")))
    
    print(f"## Benchmark Summary: {results_dir}\n")
    
    # Header
    print(f"| Model | Total | Passed | Failed | Accuracy | Top Failure Reasons |")
    print(f"|---|---|---|---|---|---|")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"| {json_file.stem} | Error reading file | - | - | - | - |")
                continue
                
        total = len(data)
        passed = sum(1 for item in data if item.get('correct', False))
        failed = total - passed
        accuracy = (passed / total * 100) if total > 0 else 0
        
        # Analyze failures
        failure_reasons = []
        for item in data:
            if not item.get('correct', False):
                error = item.get('error')
                if error:
                    # Simplify common errors
                    if "The truth value of a Series is ambiguous" in error:
                        reason = "ValueError: Ambiguous Series truth value"
                    elif "name" in error and "is not defined" in error:
                        reason = "NameError: Variable not defined"
                    elif "No code extracted" in error:
                        reason = "No code extracted"
                    elif "API error" in error:
                        reason = "API Error"
                    else:
                        # Truncate long errors
                        reason = (error[:50] + '...') if len(error) > 50 else error
                else:
                    reason = "Incorrect Result"
                failure_reasons.append(reason)
        
        # Top 3 failure reasons
        reason_counts = Counter(failure_reasons)
        top_reasons = ", ".join([f"{r} ({c})" for r, c in reason_counts.most_common(3)])
        
        model_name = json_file.stem.replace('_', '/')
        print(f"| {model_name} | {total} | {passed} | {failed} | {accuracy:.1f}% | {top_reasons} |")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <results_dir>")
        sys.exit(1)
    
    summarize_results(sys.argv[1])
