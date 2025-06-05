from pathlib import Path
from huggingface_hub import upload_folder, login
from config import (
    MODELS_DIR,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

def upload_model():
    """Upload model to HuggingFace Hub with environment-specific error handling."""
    print(f"Uploading model in {ENVIRONMENT} environment...")
    
    try:
        # Load token from environment variable
        import os
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        login(token=hf_token)
        print("[✓] HuggingFace authentication successful")
        
        model_path = MODELS_DIR / "zhbert-finetuned-v2"
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        print(f"↑ Uploading model from {model_path}...")
        upload_folder(
            folder_path=str(model_path),
            path_in_repo="",
            repo_id="eatingchew/zhbert_finetuned-v2",
            repo_type="model"
        )
        print("✅ Model uploaded successfully")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        if is_colab:
            print("Please ensure you have mounted Google Drive and the model is in the correct location.")
        elif is_kaggle:
            print("Please ensure you have added the model to your Kaggle dataset.")
        else:
            print("Please check if the model directory exists and contains the required files.")
        raise

if __name__ == "__main__":
    upload_model()