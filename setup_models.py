from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from huggingface_hub import login
import os

# ⬇️ 先讀取 .env 裡面的 HF_TOKEN（如果存在的話）
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

# ⬇️ 設定要下載的模型
MODEL_CONFIGS = {
    "zhbert": {
        "hf_id": "hfl/chinese-roberta-wwm-ext",
        "model_cls": AutoModelForSequenceClassification
    },
    "labse": {
        "hf_id": "sentence-transformers/LaBSE",
        "model_cls": AutoModel
    },
    "cross_encoder": {
        "hf_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_cls": AutoModelForSequenceClassification
    },
    "zhbert_finetuned": {   # 👈 新增你上傳到 Hugging Face 的 fine-tuned model
        "hf_id": "eatingchew/zhbert_finetuned",
        "model_cls": AutoModelForSequenceClassification
    }
}

BASE_DIR = Path("models")

def download_model(name, config):
    save_path = BASE_DIR / name
    if save_path.exists() and any(save_path.iterdir()):
        print(f"[✓] {name} already exists, skipping.")
        return
    print(f"↓ Downloading {name} from {config['hf_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
    model = config["model_cls"].from_pretrained(config["hf_id"])
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"[✓] Saved to {save_path}")

if __name__ == "__main__":
    for name, cfg in MODEL_CONFIGS.items():
        download_model(name, cfg)