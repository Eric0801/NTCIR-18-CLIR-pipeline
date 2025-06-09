#config.py - Complete Fixed Version

import os
from pathlib import Path
from typing import Dict, Optional
import sys

# -----------------------------
# Environment Detection
# -----------------------------
def detect_environment() -> str:
    """Detect the current running environment."""
    # Check for Colab
    if os.path.exists('/content'):
        return 'colab'
    # Check for Kaggle
    if os.path.exists('/kaggle'):
        return 'kaggle'
    # Check for Jupyter
    if 'jupyter' in sys.modules or 'ipykernel' in sys.modules:
        return 'jupyter'
    # Default to local
    return 'local'

def get_project_root() -> Path:
    """Get the project root directory based on environment."""
    env = detect_environment()
    
    # Try to find the project root in different ways
    possible_roots = []
    
    if env == 'colab':
        possible_roots = [
            Path('/content/NTCIR-18-CLIR-pipeline-team6939'),
            Path('/content/NTCIR-18-CLIR-pipeline'),
            Path.cwd()  # Current working directory
        ]
    elif env == 'kaggle':
        possible_roots = [
            Path('/kaggle/working/NTCIR-18-CLIR-pipeline-team6939'),
            Path('/kaggle/working/NTCIR-18-CLIR-pipeline'),
            Path.cwd()
        ]
    else:  # local or jupyter
        current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
        possible_roots = [
            current_file.parent,  # Directory containing config.py
            Path.cwd(),  # Current working directory
            Path.cwd().parent  # Parent of current directory
        ]
    
    # Find the first valid project root
    for root in possible_roots:
        if root.exists():
            # Check for key indicators of project root
            indicators = ['src', 'data', 'models', '.git', 'setup.py', 'README.md', 'run_all_retrievals.py']
            if any((root / indicator).exists() for indicator in indicators):
                return root
    
    # If no valid root found, use the current directory
    print(f"⚠️ Could not find project root, using current directory: {Path.cwd()}")
    return Path.cwd()

# -----------------------------
# Base Configuration
# -----------------------------
PROJECT_ROOT = get_project_root()
ENVIRONMENT = detect_environment()

# Print environment info for debugging
print(f"🌍 Running in {ENVIRONMENT} environment")
print(f"📁 Project root: {PROJECT_ROOT}")

# -----------------------------
# Directory Structure
# -----------------------------
# Main directories
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
PDFS_DIR = PROJECT_ROOT / 'pdfs'
SRC_DIR = PROJECT_ROOT / 'src'

# Subdirectories
RUNS_DIR = OUTPUTS_DIR / 'runs'
CACHE_DIR = OUTPUTS_DIR / 'cache'
LOGS_DIR = OUTPUTS_DIR / 'logs'
METRICS_DIR = OUTPUTS_DIR / 'metrics'  # Added this
PDFS_FINANCE_DIR = PDFS_DIR / 'finance'
PDFS_INSURANCE_DIR = PDFS_DIR / 'insurance'

# -----------------------------
# File Paths
# -----------------------------
# Data files
PASSAGE_PATH = OUTPUTS_DIR / 'structured_passages.jsonl'
QUERY_PATH = DATA_DIR / 'translated_query.json'
GROUND_TRUTH_PATH = DATA_DIR / 'ground_truths_example.json'
USER_DICT_PATH = DATA_DIR / 'userdict.txt'

# Analysis files (Added these)
ERROR_ANALYSIS_PATH = DATA_DIR / 'Error_analysis.csv'
TOP1_SUMMARY_PATH = DATA_DIR / 'top1_summary.csv'

# Model directories
ZHBERT_MODEL_DIR = MODELS_DIR / 'zhbert_finetuned-v2'
LABSE_MODEL_DIR = MODELS_DIR / 'labse'
CROSS_ENCODER_MODEL_DIR = MODELS_DIR / 'cross_encoder'

# Alternative model directories (for compatibility)
ZHBERT_MODEL_DIR_ALT = MODELS_DIR / 'zhbert'
LABSE_MODEL_DIR_ALT = MODELS_DIR / 'labse'
CROSS_ENCODER_MODEL_DIR_ALT = MODELS_DIR / 'cross_encoder'

# -----------------------------
# Output Paths (Standardized)
# -----------------------------
# Define model names
MODEL_NAMES = {
    'bm25': 'BM25',
    'dense': 'DenseEncoder',
    'cross': 'CrossEncoder',
    'zhbert': 'ZhBERT',
    'bm25_only_query': 'BM25_Query_Only',
    'bm25_only_query_zh_nmt': 'BM25_Query_ZH_NMT',
    'bm25_rerank_query': 'BM25_Rerank_Query',
    'bm25_rerank_query_zh_nmt': 'BM25_Rerank_ZH_NMT',
    'dense_dual_encoder': 'Dense_Dual_Encoder',
    'cross_encoder': 'Cross_Encoder'
}

# Standardized output paths
def get_output_path(model_name: str) -> Path:
    """Get standardized output path for a model."""
    return RUNS_DIR / f"{model_name}.jsonl"

# Model-specific output paths (Extended)
OUTPUT_PATHS = {
    'bm25': get_output_path('bm25'),
    'dense': get_output_path('dense'),
    'cross': get_output_path('cross'),
    'zhbert': get_output_path('zhbert'),
    'bm25_only_query': get_output_path('bm25_only_query'),
    'bm25_only_query_zh_nmt': get_output_path('bm25_only_query_zh_nmt'),
    'bm25_rerank_query': get_output_path('bm25_rerank_query'),
    'bm25_rerank_query_zh_nmt': get_output_path('bm25_rerank_query_zh_nmt'),
    'dense_dual_encoder': get_output_path('dense_dual_encoder'),
    'cross_encoder': get_output_path('cross_encoder')
}

# Evaluation outputs
EVALUATION_OUTPUT = OUTPUTS_DIR / 'evaluation_summary.csv'
RETRIEVAL_RANKINGS = RUNS_DIR / 'retrieval_rankings.json'

# -----------------------------
# Environment-specific Settings
# -----------------------------
ENV_SETTINGS: Dict[str, Dict] = {
    'colab': {
        'use_gpu': True,
        'batch_size': 64,
        'max_length': 512,
        'cache_dir': '/content/cache',
        'base_path': '/content/NTCIR-18-CLIR-pipeline-team6939',
        'timeout': 3600
    },
    'kaggle': {
        'use_gpu': True,
        'batch_size': 32,
        'max_length': 384,
        'cache_dir': '/kaggle/working/cache',
        'base_path': '/kaggle/working/NTCIR-18-CLIR-pipeline-team6939',
        'timeout': 7200
    },
    'jupyter': {
        'use_gpu': False,
        'batch_size': 16,
        'max_length': 256,
        'cache_dir': str(CACHE_DIR),
        'base_path': str(PROJECT_ROOT),
        'timeout': 1800
    },
    'local': {
        'use_gpu': False,
        'batch_size': 16,
        'max_length': 256,
        'cache_dir': str(CACHE_DIR),
        'base_path': str(PROJECT_ROOT),
        'timeout': 1800
    }
}

# Get current environment settings
CURRENT_ENV_SETTINGS = ENV_SETTINGS[ENVIRONMENT]

# -----------------------------
# Utility Functions
# -----------------------------
def get_cache_path(model_name: str) -> Path:
    """Get cache path for a specific model."""
    return CACHE_DIR / f'translated_cache_{model_name.replace("/", "_").replace("-", "_")}.json'

def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not create directory {path}: {e}")

def get_model_path(model_type: str, version: Optional[str] = None) -> Path:
    """Get path for a specific model type and version."""
    base_path = MODELS_DIR / model_type
    if version:
        return base_path / version
    return base_path

def verify_paths() -> bool:
    """Verify that all necessary paths exist and are accessible."""
    required_dirs = [
        DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PDFS_DIR, SRC_DIR,
        RUNS_DIR, CACHE_DIR, LOGS_DIR, METRICS_DIR
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not directory.exists():
            print(f"⚠️ Directory does not exist, creating: {directory}")
            try:
                ensure_dir(directory)
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Failed to create directories: {missing_dirs}")
        return False
    
    return True

def check_file_exists(file_path: Path, description: str = "") -> bool:
    """Check if a file exists and print status."""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"⚠️ {description} not found: {file_path}")
        return False

# -----------------------------
# Initialize Directories
# -----------------------------
# Create all necessary directories
directories_to_create = [
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PDFS_DIR, SRC_DIR,
    RUNS_DIR, CACHE_DIR, LOGS_DIR, METRICS_DIR,
    PDFS_FINANCE_DIR, PDFS_INSURANCE_DIR
]

for directory in directories_to_create:
    ensure_dir(directory)

# Verify paths
paths_ok = verify_paths()

# -----------------------------
# Logging Configuration
# -----------------------------
LOG_FILE = LOGS_DIR / f'pipeline_{ENVIRONMENT}.log'

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_CONFIGS = {
    'zhbert': {
        'hf_id': 'hfl/chinese-roberta-wwm-ext',
        'model_cls': 'AutoModelForSequenceClassification',
        'max_length': CURRENT_ENV_SETTINGS['max_length'],
        'batch_size': CURRENT_ENV_SETTINGS['batch_size'],
        'model_path': ZHBERT_MODEL_DIR,
        'model_path_alt': ZHBERT_MODEL_DIR_ALT,
        'output_path': OUTPUT_PATHS['zhbert'],
        'is_classifier': True
    },
    'labse': {
        'hf_id': 'sentence-transformers/LaBSE',
        'model_cls': 'sentence-transformer',
        'max_length': 128,
        'batch_size': CURRENT_ENV_SETTINGS['batch_size'],
        'model_path': LABSE_MODEL_DIR,
        'model_path_alt': LABSE_MODEL_DIR_ALT,
        'output_path': OUTPUT_PATHS['dense'],
        'is_classifier': False
    },
    'cross_encoder': {
        'hf_id': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'model_cls': 'AutoModelForSequenceClassification',
        'max_length': CURRENT_ENV_SETTINGS['max_length'],
        'batch_size': max(CURRENT_ENV_SETTINGS['batch_size'] // 4, 1),
        'model_path': CROSS_ENCODER_MODEL_DIR,
        'model_path_alt': CROSS_ENCODER_MODEL_DIR_ALT,
        'output_path': OUTPUT_PATHS['cross'],
        'is_classifier': True
    }
}

# -----------------------------
# Final Status Report
# -----------------------------
print(f"\n✅ Configuration initialization completed!")
print(f"   - Environment: {ENVIRONMENT}")
print(f"   - Project Root: {PROJECT_ROOT}")
print(f"   - GPU Available: {CURRENT_ENV_SETTINGS['use_gpu']}")
print(f"   - Batch Size: {CURRENT_ENV_SETTINGS['batch_size']}")
print(f"   - Paths verified: {'✅' if paths_ok else '❌'}")

# Check key files
print(f"\n📂 Key Directories:")
print(f"   - DATA_DIR: {DATA_DIR}")
print(f"   - MODELS_DIR: {MODELS_DIR}")
print(f"   - OUTPUTS_DIR: {OUTPUTS_DIR}")
print(f"   - RUNS_DIR: {RUNS_DIR}")
print(f"   - METRICS_DIR: {METRICS_DIR}")

print(f"\n📄 Key File Paths:")
print(f"   - GROUND_TRUTH_PATH: {GROUND_TRUTH_PATH}")
print(f"   - ERROR_ANALYSIS_PATH: {ERROR_ANALYSIS_PATH}")
print(f"   - TOP1_SUMMARY_PATH: {TOP1_SUMMARY_PATH}")

print(f"\n🎯 Configuration ready for use!")
