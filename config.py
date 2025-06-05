#config.py

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
    if 'jupyter' in sys.modules:
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
            Path('/content/NTCIR-18-CLIR-pipeline'),
            Path('/content/NTCIR-18-CLIR-pipeline-team6939'),
            Path.cwd()  # Current working directory
        ]
    elif env == 'kaggle':
        possible_roots = [
            Path('/kaggle/working/NTCIR-18-CLIR-pipeline'),
            Path('/kaggle/working/NTCIR-18-CLIR-pipeline-team6939'),
            Path.cwd()
        ]
    else:  # local or jupyter
        possible_roots = [
            Path(__file__).parent,  # Directory containing config.py
            Path.cwd(),  # Current working directory
            Path.cwd().parent  # Parent of current directory
        ]
    
    # Find the first valid project root
    for root in possible_roots:
        if root.exists() and (root / 'src').exists():
            return root
    
    # If no valid root found, use the current directory
    return Path.cwd()

# -----------------------------
# Base Configuration
# -----------------------------
PROJECT_ROOT = get_project_root()
ENVIRONMENT = detect_environment()

# Print environment info for debugging
print(f"Running in {ENVIRONMENT} environment")
print(f"Project root: {PROJECT_ROOT}")

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

# Model directories
ZHBERT_MODEL_DIR = MODELS_DIR / 'zhbert_finetuned-v2'
LABSE_MODEL_DIR = MODELS_DIR / 'labse'
CROSS_ENCODER_MODEL_DIR = MODELS_DIR / 'cross_encoder'

# -----------------------------
# Output Paths (Standardized)
# -----------------------------
# Define model names
MODEL_NAMES = {
    'bm25': 'BM25',
    'dense': 'DenseEncoder',
    'cross': 'CrossEncoder',
    'zhbert': 'ZhBERT'
}

# Standardized output paths
def get_output_path(model_name: str) -> Path:
    """Get standardized output path for a model."""
    return RUNS_DIR / f"{model_name}_results.jsonl"

# Model-specific output paths
OUTPUT_PATHS = {
    'bm25': get_output_path('bm25'),
    'dense': get_output_path('dense'),
    'cross': get_output_path('cross'),
    'zhbert': get_output_path('zhbert')
}

# Evaluation output
EVALUATION_OUTPUT = OUTPUTS_DIR / 'evaluation_summary.csv'

# -----------------------------
# Environment-specific Settings
# -----------------------------
ENV_SETTINGS: Dict[str, Dict] = {
    'colab': {
        'use_gpu': True,
        'batch_size': 64,
        'cache_dir': '/content/cache',
        'base_path': '/content/NTCIR-18-CLIR-pipeline'
    },
    'kaggle': {
        'use_gpu': True,
        'batch_size': 32,
        'cache_dir': '/kaggle/working/cache',
        'base_path': '/kaggle/working/NTCIR-18-CLIR-pipeline'
    },
    'jupyter': {
        'use_gpu': False,
        'batch_size': 16,
        'cache_dir': str(CACHE_DIR),
        'base_path': str(PROJECT_ROOT)
    },
    'local': {
        'use_gpu': False,
        'batch_size': 16,
        'cache_dir': str(CACHE_DIR),
        'base_path': str(PROJECT_ROOT)
    }
}

# Get current environment settings
CURRENT_ENV_SETTINGS = ENV_SETTINGS[ENVIRONMENT]

# -----------------------------
# Utility Functions
# -----------------------------
def get_cache_path(model_name: str) -> Path:
    """Get cache path for a specific model."""
    return CACHE_DIR / f'translated_cache_{model_name.replace("/", "_")}.json'

def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)

def get_model_path(model_type: str, version: Optional[str] = None) -> Path:
    """Get path for a specific model type and version."""
    base_path = MODELS_DIR / model_type
    if version:
        return base_path / version
    return base_path

def verify_paths() -> bool:
    """Verify that all necessary paths exist and are accessible."""
    required_dirs = [
        DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PDFS_DIR,
        RUNS_DIR, CACHE_DIR, LOGS_DIR
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist. Creating...")
            ensure_dir(directory)
    
    return True

# -----------------------------
# Initialize Directories
# -----------------------------
# Create all necessary directories
for directory in [
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PDFS_DIR,
    RUNS_DIR, CACHE_DIR, LOGS_DIR,
    PDFS_FINANCE_DIR, PDFS_INSURANCE_DIR
]:
    ensure_dir(directory)

# Verify paths
verify_paths()

# -----------------------------
# Logging Configuration
# -----------------------------
LOG_FILE = LOGS_DIR / f'pipeline_{ENVIRONMENT}.log'

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_CONFIGS = {
    'zhbert': {
        'max_length': 512,
        'batch_size': CURRENT_ENV_SETTINGS['batch_size'],
        'model_path': ZHBERT_MODEL_DIR,
        'output_path': OUTPUT_PATHS['zhbert']
    },
    'labse': {
        'max_length': 128,
        'batch_size': CURRENT_ENV_SETTINGS['batch_size'],
        'model_path': LABSE_MODEL_DIR,
        'output_path': OUTPUT_PATHS['dense']
    },
    'cross_encoder': {
        'max_length': 512,
        'batch_size': CURRENT_ENV_SETTINGS['batch_size'],
        'model_path': CROSS_ENCODER_MODEL_DIR,
        'output_path': OUTPUT_PATHS['cross']
    }
}
