# utils/file_utils.py
import os
import tempfile
import shutil
from utils.logging_utils import logger

def get_temp_file_path(prefix="lg_", suffix=".tmp"):
    """Get a path for a temporary file."""
    return os.path.join(tempfile.gettempdir(), f"{prefix}{tempfile.mktemp()}{suffix}")

def cleanup_temp_file(file_path):
    """Safely remove a temporary file."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error cleaning up temp file {file_path}: {e}")
            return False
    return True

def ensure_directory(directory):
    """Ensure a directory exists, creating it if needed."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False