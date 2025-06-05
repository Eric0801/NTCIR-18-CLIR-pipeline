#fine-tuned Chinese BERT reranker 這個是用來fine-tune reranker 的程式
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from config import QUERY_PATH, PASSAGE_PATH, GROUND_TRUTH_PATH, ZHBERT_MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入資料
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = {str(q["qid"]): q for q in json.load(f)}

with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = {p["pid"]: p for p in map(json.loads, f)}

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    ground_truths = {str(g["qid"]): str(g["retrieve"]) for g in raw_data["ground_truths"]}
    
# 建立 Dataset
class QPDataset(Dataset):
    def __init__(self, queries, passages, ground_truths, tokenizer, max_len=512):
        self.pairs = []
        self.tokenizer = tokenizer
        for qid, query_obj in queries.items():
            query_text = query_obj["query"]
            pos_pid = ground_truths.get(qid)
            if pos_pid:
                passage_text = passages.get(pos_pid, {}).get("text", "")
                self.pairs.append((query_text, passage_text))
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, passage = self.pairs[idx]
        encoded = self.tokenizer(query, passage, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}, torch.tensor(1)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
dataset = QPDataset(queries, passages, ground_truths, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 初始化模型
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 訓練
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# 儲存模型
os.makedirs(ZHBERT_MODEL_DIR, exist_ok=True)
model.save_pretrained(ZHBERT_MODEL_DIR)
tokenizer.save_pretrained(ZHBERT_MODEL_DIR)

print(f"[✓] Fine-tuned model saved to {ZHBERT_MODEL_DIR}")