
# 📘 Cross-Lingual IR for Traditional Chinese Financial Documents

This is the official demo package for our NTCIR-18 AI Cup submission:

> **"Translation or Multilingual Retrieval? Evaluating Cross-Lingual Search Strategies for Traditional Chinese Financial Documents"**

---

## 📦 Folder Structure

```
clir_pipeline/
├── CLIR_Reviewer_Demo_Full.ipynb   ← 📘 Main demo notebook
├── setup_models.py                 ← 🔧 Auto-download models to /models
├── run_all_retrievals.py          ← 🔁 Run 4 retrieval models with timing
│
├── models/                         ← 🧠 Local models (auto-downloaded)
│   ├── zhbert/                     ← Chinese BERT reranker
│   ├── labse/                      ← LaBSE dense encoder
│   └── cross_encoder/             ← Multilingual cross encoder
│
├── data/                           ← 📂 Query + Ground truth + Dictionary
│   ├── translated_query.json
│   ├── ground_truths_example.json
│   └── userdict.txt               ← (optional) for jieba customization
│
├── outputs/
│   ├── structured_passages.jsonl  ← 📄 Extracted paragraphs from PDFs
│   ├── runs/
│   │   ├── bm25_only.jsonl
│   │   ├── bm25_rerank.jsonl
│   │   ├── dense_dual_encoder.jsonl
│   │   └── cross_encoder.jsonl
│   └── evaluation_summary_all.csv
│
├── pdfs/                          ← 📚 Source documents
│   ├── faq/                       ← Contains queries and pid_map.json
│   ├── finance/                   ← ~1000 PDFs
│   └── insurance/                 ← ~643 PDFs
│
└── src/
    ├── retrievers/
    │   ├── bm25_only.py
    │   └── dual_encoder_dense.py
    └── reranker/
        ├── reranker_zhbert.py
        └── cross_encoder_multilingual.py
```

---

## 🚀 How to Run in Colab

```python
# Step 0: Download models (only first time)
!python setup_models.py

# Step 1: Run 4 retrieval models and generate `.jsonl` results
from run_all_retrievals import run_all_retrievals
run_all_retrievals()

# Step 2: Evaluate
from evaluation_summary import evaluate_all_models

# Step 3: Analyze translation impact
from translate_error_analysis import extract_translation_impact
```

---

## 📊 Evaluated Metrics

- MRR@10 / NDCG@10 / Recall@100
- Runtime (logged per model)
- Translation error sensitivity analysis

---

## ✍️ Reviewer's Comments Addressed

- ✅ BM25 baseline added
- ✅ Runtime comparison included
- ✅ Cross-encoder added
- ✅ System flowchart (available in `assets/`)
- ✅ Translation error cases analyzed
- ✅ Citation style fixed
- ✅ Team name removed in paper

