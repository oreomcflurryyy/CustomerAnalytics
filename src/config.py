from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------

DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# -----------------------------
# ML Configuration
# -----------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20