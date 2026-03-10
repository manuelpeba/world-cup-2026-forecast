from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Models
MODELS_DIR = ARTIFACTS_DIR / "models"