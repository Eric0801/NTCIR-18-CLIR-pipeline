from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import os
from config import (
    MODELS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT,
    MODEL_CONFIGS as CONFIG_MODEL_CONFIGS
)

def setup_huggingface():
    """Setup HuggingFace authentication."""
    try:
        if os.path.exists(".env"):
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                print("[✓] HuggingFace private token loaded.")
            else:
                print("⚠️ .env file found, but HF_TOKEN is empty. Will try anonymous access.")
        else:
            print("⚠️ No .env file found. Will try anonymous access.")
    except Exception as e:
        print(f"❌ Error setting up HuggingFace authentication: {e}")
        if is_colab:
            print("Please ensure you have mounted Google Drive and the .env file is in the correct location.")
        elif is_kaggle:
            print("Please ensure you have added the .env file to your Kaggle dataset.")
        else:
            print("Please check if the .env file exists and contains a valid HF_TOKEN.")
        raise

# ⬇️ 模型設定
MODEL_CONFIGS = {
    "zhbert": {
        "hf_id": "hfl/chinese-roberta-wwm-ext",
        "model_cls": AutoModelForSequenceClassification
    },
    "labse": {
        "hf_id": "sentence-transformers/LaBSE",
        "model_cls": "sentence-transformer"  # 👈 特別處理
    },
    "cross_encoder": {
        "hf_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_cls": AutoModelForSequenceClassification
    },
    "zhbert_finetuned-v2": {
        "hf_id": "eatingchew/zhbert_finetuned-v2",
        "model_cls": AutoModelForSequenceClassification
    }
}

def download_model(name, config):
    """Download and save a model."""
    save_path = MODELS_DIR / name
    if save_path.exists() and any(save_path.iterdir()):
        print(f"[✓] {name} already exists, skipping.")
        return

    print(f"↓ Downloading {name} from {config['hf_id']}...")
    try:
        ensure_dir(save_path)
    except Exception as e:
        print(f"❌ Error creating model directory: {e}")
        if is_colab:
            print("Please ensure you have write permissions in your Google Drive.")
        elif is_kaggle:
            print("Please ensure you have write permissions in your Kaggle workspace.")
        raise

    try:
        if config["model_cls"] == "sentence-transformer":
            model = SentenceTransformer(config["hf_id"])
            model.save(str(save_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
            model = config["model_cls"].from_pretrained(config["hf_id"])
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)

        print(f"[✓] Saved to {save_path}")
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        if is_colab:
            print("Please ensure you have mounted Google Drive and sufficient space.")
        elif is_kaggle:
            print("Please ensure you have sufficient space in your Kaggle workspace.")
        else:
            print("Please check your internet connection and available disk space.")
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path)
        raise

def main():
    """Main function to set up all models."""
    print(f"Setting up models in {ENVIRONMENT} environment...")
    try:
        setup_huggingface()
        ensure_dir(MODELS_DIR)
        
        for name, cfg in CONFIG_MODEL_CONFIGS.items():
            try:
                download_model(name, cfg)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                if is_colab:
                    print("Please ensure you have mounted Google Drive and sufficient space.")
                elif is_kaggle:
                    print("Please ensure you have sufficient space in your Kaggle workspace.")
                else:
                    print("Please check your internet connection and available disk space.")
                continue
    except Exception as e:
        print(f"❌ Error during model setup: {e}")
        raise

if __name__ == "__main__":
    main()