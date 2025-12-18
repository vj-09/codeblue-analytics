import subprocess
import sys
import argparse
import time
from pathlib import Path

# Models to evaluate
MODELS = [
    # Open Weights / Cheap
    ("qwen/qwen-2.5-72b-instruct", "prime"),
    ("x-ai/grok-4-fast", "prime"),
    ("google/gemini-2.5-flash", "prime"),
    ("deepseek/deepseek-chat", "prime"),
    ("prime-intellect/intellect-3", "prime"),
    
    # Expensive / SOTA
    ("qwen/qwen3-max", "prime"),
    ("anthropic/claude-opus-4.5", "local"),
]

SPLIT = "bank/v2_hard"
OUTPUT_DIR = Path("results/bank_v2_hard")

def run_eval_parallel(models, num_examples=-1):
    processes = []
    
    print(f"Starting PARALLEL benchmark suite for {len(models)} models on {SPLIT}...")
    
    for model, backend in models:
        # Sanitize model name for filename
        safe_name = model.replace("/", "_")
        output_file = OUTPUT_DIR / f"{safe_name}.json"
        
        cmd = [
            sys.executable, "scripts/eval_runner.py",
            "--split", SPLIT,
            "--model", model,
            "--backend", backend,
            "--output", str(output_file)
        ]
        
        if num_examples != -1:
            cmd.extend(["--num-examples", str(num_examples)])
            
        print(f"🚀 Launching {model}...")
        # Start process without waiting
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((model, p, output_file))
    
    print(f"\nAll {len(processes)} processes launched. Waiting for completion...\n")
    
    # Monitor processes
    completed = 0
    while completed < len(processes):
        for i, (model, p, outfile) in enumerate(processes):
            if p.poll() is not None: # Process finished
                # Check if we already handled this one
                if p.returncode is not None and p.stdout: 
                    # Consume output to avoid blocking (though we used PIPE)
                    stdout, stderr = p.communicate()
                    
                    if p.returncode == 0:
                        print(f"✅ Finished {model}. Results saved to {outfile}")
                    else:
                        print(f"❌ Failed {model}. Exit code: {p.returncode}")
                        print(f"Error Output:\n{stderr}")
                    
                    # Mark as handled by setting stdout to None
                    p.stdout = None 
                    completed += 1
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite in parallel")
    parser.add_argument('--test', action='store_true', help='Run only 1 example per model for testing')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    num_examples = 1 if args.test else -1
    
    run_eval_parallel(MODELS, num_examples)
        
    print("\nBenchmark suite completed.")

if __name__ == "__main__":
    main()
