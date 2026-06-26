# src/config.py

from pathlib import Path

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

# Dataset
DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Random seed
RANDOM_STATE = 42

# Test split
TEST_SIZE = 0.2