import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
from config import QUERY_PATH, GROUND_TRUTH_PATH, PASSAGE_PATH, ZHBERT_MODEL_DIR

# fine-tune parameters
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 512

device = torch.device("mps" if torch.cuda.is_available() else "cpu") #replace cuda with mps

# ===== 資料載入 =====
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = {str(q["qid"]): q for q in json.load(f)}

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    gts = json.load(f)["ground_truths"]
    ground_truths = {str(g["qid"]): str(g["retrieve"]) for g in gts}

with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = {str(json.loads(line)["pid"]): json.loads(line)["text"] for line in f if "text" in json.loads(line)}

# ===== Dataset =====
class QPDataset(Dataset):
    def __init__(self, queries, ground_truths, passages, tokenizer, max_len=512):
        self.pairs = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for qid, query_obj in queries.items():
            query_text = query_obj["query"]
            pos_pid = ground_truths.get(qid)
            source_pids = query_obj.get("source", [])

            # 正例
            if pos_pid and pos_pid in passages:
                self.pairs.append((query_text, passages[pos_pid], 1))

            # 負例：從 source 中排除正解 pid
            for pid in source_pids:
                pid = str(pid)
                if pid != pos_pid and pid in passages:
                    self.pairs.append((query_text, passages[pid], 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, passage, label = self.pairs[idx]
        encoded = self.tokenizer(query, passage, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}, torch.tensor(label, dtype=torch.long)

# ===== 建立 dataset & dataloader =====
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
dataset = QPDataset(queries, ground_truths, passages, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== 初始化模型 =====
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)

# ===== 開始訓練 =====
model.train()
for epoch in range(EPOCHS):
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

# ===== 儲存模型與 tokenizer =====
os.makedirs(ZHBERT_MODEL_DIR, exist_ok=True)
model.save_pretrained(ZHBERT_MODEL_DIR)
tokenizer.save_pretrained(ZHBERT_MODEL_DIR)
print(f"[✓] Fine-tuned model saved to {ZHBERT_MODEL_DIR}")
