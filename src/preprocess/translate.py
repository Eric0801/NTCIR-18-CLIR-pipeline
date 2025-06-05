from transformers import MarianMTModel, MarianTokenizer
import torch
import json
from tqdm import tqdm
from opencc import OpenCC
from pathlib import Path
from config import (
    QUERY_PATH,
    OUTPUTS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

def load_model():
    """Load translation model with environment-specific error handling."""
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model_name, tokenizer, model
    except Exception as e:
        if is_colab:
            print(f"Error loading model in Colab: {str(e)}")
            print("Please ensure you have mounted Google Drive and the model is in the correct location.")
        elif is_kaggle:
            print(f"Error loading model in Kaggle: {str(e)}")
            print("Please ensure you have added the model dataset and it's in the correct location.")
        else:
            print(f"Error loading model: {str(e)}")
        raise

def load_queries():
    """Load queries with environment-specific error handling."""
    try:
        with open(QUERY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if is_colab:
            print(f"Query file not found at {QUERY_PATH}. Please ensure you have mounted Google Drive and the file is in the correct location.")
        elif is_kaggle:
            print(f"Query file not found at {QUERY_PATH}. Please ensure you have added the dataset and the file is in the correct location.")
        else:
            print(f"Query file not found at {QUERY_PATH}. Please check if the file exists.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {QUERY_PATH}: {str(e)}")
        raise

def load_cache(cache_path):
    """Load translation cache if it exists."""
    try:
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading cache: {str(e)}")
        return {}

def translate_with_nmt(query_en, tokenizer, model, cc, translated_cache, convert_to_traditional=True):
    """Translate query with NMT model."""
    if query_en in translated_cache:
        return translated_cache[query_en]
    try:
        inputs = tokenizer(query_en, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512, num_beams=5)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        if convert_to_traditional:
            result = cc.convert(result)
        translated_cache[query_en] = result
        return result
    except Exception as e:
        print(f"Translation error for query '{query_en}': {str(e)}")
        return ""

def main():
    """Run translation with environment-specific error handling."""
    print(f"Running translation in {ENVIRONMENT} environment...")
    ensure_dir(OUTPUTS_DIR)

    try:
        # Load model and data
        model_name, tokenizer, model = load_model()
        queries = load_queries()
        cc = OpenCC('s2t')

        # Setup cache
        cache_path = OUTPUTS_DIR / f"translated_cache_{model_name.replace('/', '_')}.json"
        translated_cache = load_cache(cache_path)

        # Translate queries
        translated_output = []
        for item in tqdm(queries, desc="Translating queries"):
            zh = translate_with_nmt(
                item["query_en"],
                tokenizer,
                model,
                cc,
                translated_cache,
                convert_to_traditional=True
            )
            item["query_zh_nmt"] = zh
            translated_output.append(item)

        # Save results
        output_path = OUTPUTS_DIR / "translated_query_nmt.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translated_output, f, indent=2, ensure_ascii=False)
            print(f"✅ Translated queries saved to: {output_path}")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(translated_cache, f, indent=2, ensure_ascii=False)
            print(f"✅ Translation cache saved to: {cache_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            if is_colab:
                print("Please ensure you have write permissions in your Google Drive.")
            elif is_kaggle:
                print("Please ensure you have write permissions in your Kaggle workspace.")
            raise

    except Exception as e:
        print(f"❌ Error during translation: {e}")
        raise

if __name__ == "__main__":
    main()