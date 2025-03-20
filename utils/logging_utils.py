# utils/logging_utils.py
import logging
import os
import sys
from config.settings import BASE_DIR

def setup_logging():
    """Configure the application logging."""
    log_dir = BASE_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use UTF-8 encoding for file handlers
    file_handler = logging.FileHandler(log_dir / "little_geeky.log", encoding='utf-8')
    
    # For console output, handle encoding issues
    class EncodingSafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # Replace problematic characters with '?'
                safe_msg = self.format(record).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
                stream = self.stream
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler,
            EncodingSafeStreamHandler()
        ]
    )
    
    # Create app logger with a plain name to avoid encoding issues in console
    logger = logging.getLogger("Little Geeky")
    return logger

# Create the logger
logger = setup_logging()