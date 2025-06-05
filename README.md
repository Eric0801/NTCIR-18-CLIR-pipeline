# 📘 Cross-Lingual IR for Traditional Chinese Financial Documents

This is the official demo package for our NTCIR-18 AI Cup submission:

> **"Translation or Multilingual Retrieval? Evaluating Cross-Lingual Search Strategies for Traditional Chinese Financial Documents"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👨‍💻 Author

This pipeline and architecture was built by [Yi-Ting, Chiu](https://github.com/Eric0801) (GitHub: [@Eric0801](https://github.com/Eric0801), also known as EatingChew). Any other repository implementing similar functionality is a descendant of this project.

## 🔑 Model and Dataset Access

Our fine-tuned Chinese-RoBERTa-wwm-ext reranker model is available on HuggingFace. The model, HF token, and dataset are available upon request for:
- 🎓 Educational purposes
- 🔬 Research use
- 🎯 Demo testing

⚠️ **Important Notice**: Commercial use of the dataset or pipeline is strictly forbidden. This includes but is not limited to:
- 🚫 Commercial product development
- 🚫 Paid services
- 🚫 Commercial research
- 🚫 Any for-profit activities

To request access, you can either:

1. **Open an issue in this repository:**
   - Specify your intended use case
   - Provide your institutional email (if applicable)
   - Confirm that your use case is non-commercial

2. **Contact the owner directly:**
   - 📧 Email: [ericchiu801@gmail.com](mailto:ericchiu801@gmail.com)
   - 💼 LinkedIn: [Yi-Ting, Chiu](https://www.linkedin.com/in/yi-ting-chiu/)

We aim to support the research community while ensuring responsible use of our resources.

## 🔒 Security and Privacy Guidelines

When using or forking this project, please ensure you do NOT share:

1. **API Keys and Tokens:**
   - 🚫 HuggingFace API tokens
   - 🚫 OpenAI API keys
   - 🚫 Any other service API credentials

2. **Environment Variables:**
   - 🚫 `.env` files
   - 🚫 Configuration files containing sensitive data
   - 🚫 Local path configurations

3. **Model Weights and Data:**
   - 🚫 Fine-tuned model weights
   - 🚫 Training datasets
   - 🚫 Evaluation datasets
   - 🚫 Any proprietary data

4. **Personal Information:**
   - 🚫 User data
   - 🚫 Personal identifiers
   - 🚫 Contact information (except for the owner's public contact)

5. **System Information:**
   - 🚫 Server configurations
   - 🚫 Database credentials
   - 🚫 Network settings

Instead, use:
- ✅ Environment variables for sensitive data
- ✅ `.gitignore` to exclude sensitive files
- ✅ Public documentation for setup instructions
- ✅ Sample configuration files with placeholder values

---

## 📦 Folder Structure

```
clir_pipeline/
├── CLIR_Reviewer_Demo_Full.ipynb   ← 📘 Main demo notebook
├── setup_models.py                 ← 🔧 Auto-download models to /models
├── run_all_retrievals.py          ← 🔁 Run 4 retrieval models with timing
├── config.py                      ← ⚙️ Centralized configuration
├── LICENSE                        ← 📄 MIT License
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

## 🌍 Environment Support

The pipeline supports multiple environments:
- 🖥️ Local development
- ☁️ Google Colab
- 📊 Kaggle

The system automatically detects the environment and adjusts paths and configurations accordingly.

### Environment Setup

1. **Local Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # or
   myenv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Google Colab**
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Install dependencies
   !pip install -r requirements.txt
   ```

3. **Kaggle**
   ```python
   # Install dependencies
   !pip install -r requirements.txt
   ```

### Configuration

The `config.py` file manages all paths and settings. It automatically:
- Detects the current environment
- Sets up appropriate paths
- Handles environment-specific requirements
- Provides error messages specific to each environment

---

## 🚀 How to Run

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

### Running Individual Components

```bash
# Run all retrievals
python run_all_retrievals.py

# Run debug pipeline
python run_all_debug.py

# Generate QR code
python QRcode.py
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
- ✅ Multi-environment support added
- ✅ Centralized configuration system
- ✅ Improved error handling and logging

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don't hold you liable.

Key features of the MIT License:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ✅ Liability limitation
- ✅ Warranty limitation

For more information about the MIT License, visit [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT).

