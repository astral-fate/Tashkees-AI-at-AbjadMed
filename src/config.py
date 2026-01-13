import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data') # Assuming data will be placed here
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Defaults
DEFAULT_MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LENGTH = 128
BATCH_SIZE = 16
SEED = 42

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
