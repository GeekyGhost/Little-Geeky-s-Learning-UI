# models/model_manager.py
import asyncio
from pathlib import Path
from utils.logging_utils import logger
from config.settings import MODELS_DIR

class ModelManager:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

    async def download_model(self, model_name: str) -> bool:
        """Download model from Ollama"""
        try:
            # Preprocess model name to strip common prefixes
            if "ollama run " in model_name:
                original_name = model_name
                model_name = model_name.replace("ollama run ", "")
                logger.info(f"Removed 'ollama run' prefix from model name: {original_name} -> {model_name}")
            
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
