# utils/settings_utils.py
import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from utils.logging_utils import logger
from config.settings import DATA_DIR

class SettingsManager:
    """Manages application settings across tabs"""
    
    GLOBAL_SETTINGS_FILE = os.path.join(DATA_DIR, "global_settings.json")
    TASK_MODELS_FILE = os.path.join(DATA_DIR, "task_models.json")
    
    @staticmethod
    def get_global_settings() -> Dict[str, Any]:
        """Get global application settings"""
        try:
            if os.path.exists(SettingsManager.GLOBAL_SETTINGS_FILE):
                with open(SettingsManager.GLOBAL_SETTINGS_FILE, 'r') as f:
                    return json.load(f)
            
            # Default settings if file doesn't exist
            return {
                "default_voice": "System Default",
                "default_speed": 1.0,
                "theme": "Blue Theme",
                "font_size": 1.0,
                "autoplay": True
            }
        except Exception as e:
            logger.error(f"Error loading global settings: {e}")
            return {
                "default_voice": "System Default",
                "default_speed": 1.0,
                "theme": "Blue Theme",
                "font_size": 1.0,
                "autoplay": True
            }
    
    @staticmethod
    def get_task_models() -> Dict[str, str]:
        """Get task-specific model configurations"""
        try:
            if os.path.exists(SettingsManager.TASK_MODELS_FILE):
                with open(SettingsManager.TASK_MODELS_FILE, 'r') as f:
                    return json.load(f)
            
            # Default models if file doesn't exist
            return {
                "text_model": "llama3.2",  # Default text generation model
                "vision_model": "llava",  # Default vision model
                "embedding_model": "all-minilm"  # Default embedding model
            }
        except Exception as e:
            logger.error(f"Error loading task models: {e}")
            return {
                "text_model": "llama3.2",
                "vision_model": "llava",
                "embedding_model": "all-minilm"
            }
    
    @staticmethod
    def get_model_for_task(task: str) -> str:
        """Get the appropriate model for a specific task"""
        task_models = SettingsManager.get_task_models()
        
        if task == "text":
            return task_models.get("text_model", "llama3.2")
        elif task == "vision":
            return task_models.get("vision_model", "llava")
        elif task == "embedding":
            return task_models.get("embedding_model", "all-minilm")
        else:
            logger.warning(f"Unknown task type: {task}, using text model")
            return task_models.get("text_model", "llama3.2")
    
    @staticmethod
    def get_voice_settings() -> Dict[str, Any]:
        """Get voice and speech settings"""
        global_settings = SettingsManager.get_global_settings()
        
        return {
            "voice": global_settings.get("default_voice", "System Default"),
            "speed": global_settings.get("default_speed", 1.0),
            "autoplay": global_settings.get("autoplay", True)
        }
    
    @staticmethod
    def save_global_settings(settings: Dict[str, Any]) -> bool:
        """Save global application settings"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(SettingsManager.GLOBAL_SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving global settings: {e}")
            return False
    
    @staticmethod
    def save_task_models(models: Dict[str, str]) -> bool:
        """Save task-specific model configurations"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(SettingsManager.TASK_MODELS_FILE, 'w') as f:
                json.dump(models, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving task models: {e}")
            return False
    
    @staticmethod
    def apply_theme_settings():
        """Generate CSS overrides based on theme settings"""
        global_settings = SettingsManager.get_global_settings()
        theme = global_settings.get("theme", "Blue Theme")
        font_size = global_settings.get("font_size", 1.0)
        
        # Theme color mappings
        theme_colors = {
            "Blue Theme": {
                "primary": "#295095",
                "secondary": "#1a365d",
                "text": "#ffffff",
                "background": "#f0f0f0"
            },
            "Dark Theme": {
                "primary": "#222222",
                "secondary": "#333333",
                "text": "#ffffff",
                "background": "#111111"
            },
            "Light Theme": {
                "primary": "#f0f0f0",
                "secondary": "#e0e0e0",
                "text": "#333333",
                "background": "#ffffff"
            },
            "High Contrast": {
                "primary": "#000000",
                "secondary": "#333333",
                "text": "#ffffff",
                "background": "#000000",
                "accent": "#ffff00"  # High-contrast yellow for accessibility
            }
        }
        
        # Get colors for selected theme
        colors = theme_colors.get(theme, theme_colors["Blue Theme"])
        
        # Generate CSS overrides
        css = f"""
        .gradio-container {{ 
            background-color: {colors["background"]};
            font-size: {font_size}em;
        }}
        
        .header {{ 
            background-color: {colors["primary"]};
            color: {colors["text"]};
        }}
        
        .gradio-button.primary {{
            background-color: {colors["primary"]};
            color: {colors["text"]};
        }}
        
        .gradio-button.secondary {{
            background-color: {colors["secondary"]};
            color: {colors["text"]};
        }}
        """
        
        # Add high contrast specific rules
        if theme == "High Contrast":
            css += f"""
            a {{
                color: {colors["accent"]};
                text-decoration: underline;
            }}
            
            input, textarea {{
                border: 2px solid {colors["accent"]};
            }}
            """
        
        return css