
import os
import subprocess

print("🧪 Running full debug pipeline for CLIR evaluation...\n")

steps = [
    ("Cross-Encoder Top-K Sanity Check", "python cross_encoder_debug_gt.py"),
    ("Evaluate All Models", "python run_eval.py")
]

for title, cmd in steps:
    print(f"🚀 {title}")
    code = subprocess.run(cmd, shell=True)
    print("")

print("✅ All debug steps completed.")
