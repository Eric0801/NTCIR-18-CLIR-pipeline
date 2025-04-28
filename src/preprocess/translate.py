from transformers import MarianMTModel, MarianTokenizer
import torch
import os
import json
from tqdm import tqdm
from opencc import OpenCC  # 新增：簡轉繁

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

cc = OpenCC('s2t') 

with open("/content/data/translated_query.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

cache_path = f"/content/outputs/translated_cache_{model_name.replace('/', '_')}.json"
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        translated_cache = json.load(f)
else:
    translated_cache = {}

def translate_with_nmt(query_en, convert_to_traditional=True):
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
        print(f"Translation error: {e}")
        return ""

translated_output = []
for item in tqdm(queries):
    zh = translate_with_nmt(item["query_en"], convert_to_traditional=True) 
    item["query_zh_nmt"] = zh
    translated_output.append(item)

os.makedirs("/content/outputs", exist_ok=True)

with open("/content/outputs/translated_query_nmt.json", "w", encoding="utf-8") as f:
    json.dump(translated_output, f, indent=2, ensure_ascii=False)

with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(translated_cache, f, indent=2, ensure_ascii=False)