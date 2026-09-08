from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Dataset files
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.csv"

# Model
MODEL_FILE = MODEL_DIR / "parking_classifier.joblib"

# Image settings
IMAGE_SIZE = (64, 64)

# Dataset limits
MAX_SAMPLES = 3000

# Train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 42


def create_directories():
    """Create required project directories."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)