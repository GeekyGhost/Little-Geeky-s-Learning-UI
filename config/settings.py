# config/settings.py
import os
from pathlib import Path

# Application settings
APP_TITLE = "Little Geeky's Learning Adventure"
APP_VERSION = "1.0.0"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"

# Create necessary directories
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# API settings
OLLAMA_API_URL = "http://127.0.0.1:11434/api"

# Audio settings
SPEECH_RATE = 175  # Default rate
DEFAULT_VOICE_GENDER = "female"  # Preferred voice gender for kids

# Document settings
MAX_IMAGE_SIZE_MB = 20