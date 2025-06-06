# 📘 Cross-Lingual IR for Traditional Chinese Financial Documents

This is the official demo package for our NTCIR-18 AI Cup submission:

> **"Translation or Multilingual Retrieval? Evaluating Cross-Lingual Search Strategies for Traditional Chinese Financial Documents"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👨‍💻 Author

This pipeline and architecture was built by [Yi-Ting, Chiu](https://github.com/Eric0801) (GitHub: [@Eric0801](https://github.com/Eric0801), also known as EatingChew). Any other repository implementing similar functionality is a descendant of this project.

## 👥 Contributors

This project also received support from [Zong-Han, Bai] (GitHub: HummerQAQ), who contributed selectively to early-stage discussions, code testing, and translation error analysis verification. While valuable, these contributions do not cover the core design, full implementation, or the NTCIR-18 paper's final revision process.

All contributions are transparently tracked in the GitHub commit and contributor history.

## 📦 Model and Dataset Access

- The fine-tuned Chinese-RoBERTa-wwm-ext reranker model is available on HuggingFace.
- **Training and evaluation datasets are NOT publicly available** due to licensing restrictions.
- For educational, research, or demo use, please contact the owner to request access (see below).
- ⚠️ **Commercial use is strictly prohibited.**

You may still use the provided code and pipeline with your own data, as long as it follows the same format as shown in the `data/` directory.

To request access, you can either:

1. **Open an issue in this repository:**
   - Specify your intended use case
   - Provide your institutional email (if applicable)
   - Confirm that your use case is non-commercial

2. **Contact the owner directly:**
   - 📧 Email: [ericchiu801@gmail.com](mailto:ericchiu801@gmail.com)
   - 💼 LinkedIn: [Yi-Ting, Chiu](https://www.linkedin.com/in/逸庭-邱/)

We aim to support the research community while ensuring responsible use of our resources.

## 🔒 Security and Privacy

**Do NOT commit or share:**
- API keys, tokens, or credentials (HuggingFace, OpenAI, etc.)
- `.env` or config files with sensitive data
- Model weights or proprietary datasets
- Personal or user data
- System credentials or server configs

Use:
- Environment variables for secrets
- `.gitignore` to exclude sensitive files
- Sample config files with placeholders
  
Instead, use:
- ✅ Environment variables for sensitive data
- ✅ `.gitignore` to exclude sensitive files
- ✅ Public documentation for setup instructions
- ✅ Sample configuration files with placeholder values

---

## 📦 Folder Structure

```
NTCIR-18-CLIR-pipeline/
│
├── CLIR_Reviewer_Demo_Full.ipynb      # Main workflow Jupyter Notebook
├── config.py                          # Path and environment auto-detection settings
├── requirements.txt                   # Python dependencies
├── run_all.sh                         # One-click pipeline shell script
├── run_all_retrievals.py              # Run all retrieval modules
├── setup_models.py                    # Script to download/setup models
├── upload_models.py                   # Script to upload models to HuggingFace
├── QRcode.py                          # QR code generation utility
│
├── data/                              # Query, annotation, and dictionary files
│   ├── translated_query.json
│   ├── ground_truths_example.json
│   ├── userdict.txt
│   └── pid_map_content.json
│
├── models/                            # Local model storage (should be .gitignored)
│   ├── zhbert_finetuned-v2/
│   ├── zhbert-finetuned-v2/
│   └── labse/
│
├── outputs/                           # Output results and intermediate files
│   ├── structured_passages.jsonl
│   └── runs/
│       └── retrieval_rankings.json
│
├── pdfs/                              # Original PDF files
│   ├── finance/
│   ├── insurance/
│   └── faq/
│
├── src/                               # Main source code
│   ├── analysis/
│   │   └── translate_error_analysis.py
│   ├── evaluation/
│   │   ├── evaluation.py
│   │   └── evaluation_summary.py
│   ├── preprocess/
│   │   └── translate.py
│   ├── reranker/
│   │   ├── bm25_finetune_reranker_dualquery.py
│   │   ├── cross_encoder_multilingual.py
│   │   ├── fine_tune_reranker.py
│   │   ├── fine_tune_reranker_v2.py
│   │   └── reranker_zhbert_dualquery.py
│   └── retrievers/
│       ├── bm25_only_dualquery.py
│       └── dual_encoder_dense.py
│
├── .gitignore
├── LICENSE
└── README.md
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

## Evaluated Metrics

- MRR@10 / NDCG@10 / Recall@100
- Runtime (logged per model)
- Translation error sensitivity analysis

---

## Reviewer's Comments Addressed

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License allows commercial and private use, modification, and distribution, provided that attribution is given.

