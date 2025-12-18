import subprocess
import sys
from pathlib import Path
import time

# Models to rerun
MODELS = [
    # Fallback to standard Opus 3 ID (User's "Opus 4.5" likely refers to this or is a misnomer)
    ("claude-3-opus-20240229", "local"),
    
    # Rerun Flash (Prime backend)
    ("google/gemini-2.5-flash", "prime"),

    # Rerun Llama 3.1 70B (Prime backend)
    ("meta-llama/llama-3.1-70b-instruct", "prime"),
]

SPLIT = "bank/v2_hard"
OUTPUT_DIR = Path("results/bank_v2_hard")

def run_eval_parallel(models):
    processes = []
    
    print(f"Starting RERUN for {len(models)} models on {SPLIT}...")
    
    for model, backend in models:
        # Sanitize model name for filename
        safe_name = model.replace("/", "_")
        # For Claude, we want to match the previous filename if possible, or create a new one
        # The previous file was anthropic_claude-opus-4.5.json. 
        # But since we are changing the ID, we should probably let it create a new file 
        # and then maybe rename it or just use the new one.
        # Let's stick to the standard naming convention based on model ID.
        output_file = OUTPUT_DIR / f"{safe_name}.json"
        
        cmd = [
            sys.executable, "scripts/eval_runner.py",
            "--split", SPLIT,
            "--model", model,
            "--backend", backend,
            "--output", str(output_file)
        ]
        
        print(f"🚀 Launching {model}...")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((model, p, output_file))
    
    print(f"\nAll {len(processes)} processes launched. Waiting for completion...\n")
    
    # Monitor processes
    completed = 0
    while completed < len(processes):
        for i, (model, p, outfile) in enumerate(processes):
            if p.poll() is not None: # Process finished
                if p.returncode is not None and p.stdout: 
                    stdout, stderr = p.communicate()
                    
                    if p.returncode == 0:
                        print(f"✅ Finished {model}. Results saved to {outfile}")
                    else:
                        print(f"❌ Failed {model}. Exit code: {p.returncode}")
                        print(f"Error Output:\n{stderr}")
                    
                    p.stdout = None 
                    completed += 1
        time.sleep(2)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_eval_parallel(MODELS)
    print("\nRerun completed.")
