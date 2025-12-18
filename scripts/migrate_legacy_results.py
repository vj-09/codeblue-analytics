import json
import argparse
from pathlib import Path
from datetime import datetime

def convert_kaggle_format(data, filename):
    """Convert Kaggle QA Miner format to EvalResult schema."""
    results = []
    model_id = data.get('model', 'unknown')
    timestamp = datetime.now().isoformat() # Legacy files might not have timestamp
    
    for item in data.get('results', []):
        # Map fields
        # Kaggle: id, question, expected_output, model_output, generated_code, status, match, error
        # Target: question_id, question, exec_success, correct, generated_code, result, expected, error, model_id, timestamp, metadata, metrics
        
        exec_success = item.get('status') == 'PASS' or item.get('status') == 'FAIL' # FAIL means executed but wrong answer
        if item.get('status') == 'ERROR':
            exec_success = False
            
        new_item = {
            "question_id": str(item.get('id')),
            "question": item.get('question'),
            "exec_success": exec_success,
            "correct": item.get('match', False),
            "generated_code": item.get('generated_code'),
            "result": item.get('model_output'),
            "expected": item.get('expected_output'),
            "error": item.get('error'),
            "model_id": model_id,
            "timestamp": timestamp,
            "metadata": {
                "source_file": filename,
                "dataset": data.get('dataset'),
                "question_set": data.get('question_set')
            },
            "metrics": {}
        }
        results.append(new_item)
    return results

def convert_manual_review_format(data, filename):
    """Convert manual_review.json format to EvalResult schema."""
    results = []
    timestamp = datetime.now().isoformat()
    
    for item in data.get('questions', []):
        q_id = item.get('id')
        question = item.get('question')
        expected = item.get('expected_output')
        dataset = item.get('dataset')
        
        # Metadata from the question
        base_metadata = {
            "source_file": filename,
            "dataset": dataset,
            "grade": item.get('grade'),
            "manual_review": item.get('manual_review')
        }
        
        # Iterate through model outputs
        model_outputs = item.get('model_outputs', {})
        for model_id, output_data in model_outputs.items():
            # Some entries might be just a string status, skip those if they aren't dicts
            if not isinstance(output_data, dict):
                continue
                
            new_item = {
                "question_id": q_id,
                "question": question,
                "exec_success": True, # Assuming success if output exists
                "correct": output_data.get('passed', False),
                "generated_code": output_data.get('generated_code'),
                "result": output_data.get('actual_output'),
                "expected": expected,
                "error": output_data.get('error'),
                "model_id": model_id,
                "timestamp": timestamp,
                "metadata": base_metadata,
                "metrics": {}
            }
            results.append(new_item)
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Migrate legacy results to new format")
    parser.add_argument('files', nargs='+', help='Files to migrate')
    parser.add_argument('--output-dir', default='migrated_results', help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"Skipping {path}: Not found")
            continue
            
        print(f"Processing {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Detect format
            if 'results' in data and 'pass_rate' in data:
                # Kaggle QA Miner format
                converted = convert_kaggle_format(data, path.name)
            elif 'questions' in data and 'summary' in data:
                # Manual Review format
                converted = convert_manual_review_format(data, path.name)
            else:
                print(f"Unknown format for {path}")
                continue
                
            # Save
            out_name = f"migrated_{path.name}"
            out_path = out_dir / out_name
            with open(out_path, 'w') as f:
                json.dump(converted, f, indent=2)
            print(f"Saved to {out_path}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    main()
