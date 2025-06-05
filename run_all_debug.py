import subprocess
from pathlib import Path
from config import (
    ENVIRONMENT,
    LOGS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local
)

def run_debug_pipeline():
    """Run the debug pipeline for CLIR evaluation."""
    print(f"🧪 Running full debug pipeline in {ENVIRONMENT} environment...\n")
    
    try:
        ensure_dir(LOGS_DIR)
    except Exception as e:
        print(f"❌ Error creating log directory: {e}")
        if is_colab:
            print("Please ensure you have write permissions in your Google Drive.")
        elif is_kaggle:
            print("Please ensure you have write permissions in your Kaggle workspace.")
        raise

    steps = [
        ("Cross-Encoder Top-K Sanity Check", "python cross_encoder_debug_gt.py"),
        ("Evaluate All Models", "python run_eval.py")
    ]

    success = True
    for title, cmd in steps:
        print(f"🚀 {title}")
        try:
            result = subprocess.run(cmd, shell=True, check=True)
            print(f"✅ {title} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ {title} failed with error: {e}")
            if is_colab:
                print("Please ensure you have mounted Google Drive and all required files are in the correct location.")
            elif is_kaggle:
                print("Please ensure you have added all required datasets and files are in the correct location.")
            else:
                print("Please check if all required files and dependencies are properly set up.")
            success = False
        except FileNotFoundError as e:
            print(f"❌ Script not found: {e}")
            success = False
        print("")

    if success:
        print("✅ All debug steps completed successfully.")
    else:
        print("⚠️ Some debug steps failed. Check the logs for details.")
        if is_colab:
            print("Please check the logs in your Google Drive for more information.")
        elif is_kaggle:
            print("Please check the logs in your Kaggle workspace for more information.")
        else:
            print("Please check the logs in the logs directory for more information.")

if __name__ == "__main__":
    run_debug_pipeline()
