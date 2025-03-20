# ui/app.py
# Modified to remove the model dropdown from the main UI

import gradio as gr
import asyncio
import aiohttp
from utils.logging_utils import logger
from services.audio_service import AudioProcessor, VoiceRecorder
from services.document_service import DocumentProcessor
from services.ollama_service import OllamaClient
from models.model_manager import ModelManager
from models.achievements import AchievementManager
from models.user_progress import UserProgress
from ui.styles import CSS
from ui.tabs.reading_tab import ReadingTab
from ui.tabs.typing_tab import TypingTab
from ui.tabs.math_tab import MathTab
from ui.tabs.achievements_tab import AchievementsTab
from ui.tabs.settings_tab import SettingsTab
from config.settings import APP_TITLE, OLLAMA_API_URL
from utils.settings_utils import SettingsManager

class LittleGeekyApp:
    def __init__(self):
        self.ollama = OllamaClient()
        self.audio = AudioProcessor()
        self.recorder = VoiceRecorder()
        self.document_processor = DocumentProcessor()
        self.current_user = None
        self.model_manager = ModelManager()
        self.achievement_manager = AchievementManager()
        
        # Initialize tab handlers
        self.reading_tab_handler = ReadingTab(self)
        self.typing_tab_handler = TypingTab(self)
        self.math_tab_handler = MathTab(self)
        self.achievements_tab_handler = AchievementsTab(self)
        self.settings_tab_handler = SettingsTab(self)

    async def get_models(self):
        """Get available models from Ollama API"""
        try:
            # Directly fetch models to ensure we get the latest data
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_API_URL}/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'models' in data and data['models']:
                            model_list = [model['name'] for model in data['models']]
                            logger.info(f"Loaded {len(model_list)} models from Ollama")
                            return model_list
                    
                    logger.warning("No models found or API returned unexpected data")
                    return ["No models found 😕"]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return ["Please check if Ollama is running 🤔"]

    def create_interface(self):
        """Create the main application interface"""
        with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS) as iface:
            # Header section with login
            with gr.Row(elem_classes="header"):
                with gr.Column():
                    gr.Markdown("# 🌟 Little Geeky's Learning Adventure! 🚀")
                    with gr.Row():
                        username = gr.Textbox(
                            label="Username",
                            placeholder="Enter your username..."
                        )
                        login_btn = gr.Button("Login 🔑", variant="primary")
                    login_status = gr.HTML("Please login to track your progress!")
            
            async def handle_login(username):
                """Handle user login"""
                if username:
                    self.current_user = username
                    progress = UserProgress(username)
                    # Update login stats
                    progress.update_stat("logins", 1)
                    newly_earned = await self.achievement_manager.check_achievements(progress)
                    
                    message = f"Welcome back, {username}! 👋"
                    if newly_earned:
                        message += f"\n\n🎉 You've earned {len(newly_earned)} new achievement(s)! Check the Achievements tab!"
                        
                    return message
                return "Please enter a username to login."

            login_btn.click(
                fn=handle_login,
                inputs=[username],
                outputs=[login_status]
            )

            with gr.Tabs() as tabs:
                # Create tabs WITHOUT passing the model_dropdown parameter
                reading_tab = self.reading_tab_handler.create_tab(None)
                typing_tab = self.typing_tab_handler.create_tab(None)
                math_tab = self.math_tab_handler.create_tab(None)
                achievements_tab = self.achievements_tab_handler.create_tab()
                settings_tab = self.settings_tab_handler.create_tab(None)
            
            return iface