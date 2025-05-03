# ui/tabs/settings_tab.py
import gradio as gr
import os
import json
import webbrowser
import yaml
import asyncio
import aiohttp
from utils.logging_utils import logger
from config.settings import MODELS_DIR, DATA_DIR, OLLAMA_API_URL
from utils.settings_utils import SettingsManager

class SettingsTab:
    def __init__(self, app_context):
        self.app = app_context
        
    def create_tab(self, model_dropdown) -> gr.Tab:
        """Create an enhanced Settings and Model Management tab"""
        # Pre-load models data
        initial_models = self._get_model_names()
        initial_multimodal = self._get_multimodal_models(initial_models)
        initial_embeddings = self._get_embedding_models(initial_models)
        initial_model_details = self._get_model_details_sync(initial_models)
        
        with gr.Tab("Settings ⚙️") as tab:
            with gr.Tabs() as settings_tabs:
                # Global Settings Section
                with gr.Tab("Global Settings"):
                    gr.Markdown("## Global Settings")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Get current global settings
                            global_settings = SettingsManager.get_global_settings()
                            
                            # Get list of voice names
                            voice_names = [v.name for v in self.app.audio.voices] if self.app.audio.voices else ["Default System Voice"]
                            default_voice = global_settings.get("default_voice", "System Default")
                            if default_voice not in voice_names and voice_names:
                                default_voice = voice_names[0]
                            
                            # Voice and speed settings
                            global_voice = gr.Dropdown(
                                choices=voice_names,
                                label="Default Voice (used across all tabs)",
                                value=default_voice,
                                elem_classes="global-voice-selector"
                            )
                            
                            global_speed = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=global_settings.get("default_speed", 1.0),
                                step=0.1,
                                label="Default Reading Speed"
                            )
                            
                            # UI theme settings
                            theme_options = ["Blue Theme", "Dark Theme", "Light Theme", "High Contrast"]
                            theme_selector = gr.Radio(
                                choices=theme_options,
                                value=global_settings.get("theme", "Blue Theme"),
                                label="UI Theme"
                            )
                            
                            # Font size settings
                            font_size = gr.Slider(
                                minimum=1,
                                maximum=2,
                                value=global_settings.get("font_size", 1.0),
                                step=0.1,
                                label="Text Size Multiplier"
                            )
                            
                            # Autoplay settings
                            autoplay = gr.Checkbox(
                                value=global_settings.get("autoplay", True),
                                label="Enable audio autoplay"
                            )
                            
                        with gr.Column(scale=1):
                            # Preview area with initial state based on settings
                            preview_html = gr.HTML(
                                self._generate_preview_html(
                                    global_settings.get("theme", "Blue Theme"),
                                    global_settings.get("font_size", 1.0)
                                ),
                                elem_classes="settings-preview"
                            )
                            
                            # Save button for global settings
                            save_globals_btn = gr.Button("Save Global Settings", variant="primary")
                    
                    global_status = gr.Textbox(label="Status")
                    
                    # Event handlers for preview
                    def update_preview(theme, font_size):
                        return self._generate_preview_html(theme, font_size)
                    
                    theme_selector.change(
                        fn=update_preview,
                        inputs=[theme_selector, font_size],
                        outputs=[preview_html]
                    )
                    
                    font_size.change(
                        fn=update_preview,
                        inputs=[theme_selector, font_size],
                        outputs=[preview_html]
                    )
                    
                    # Save global settings function
                    def save_global_settings(voice, speed, theme, font_size, autoplay):
                        try:
                            settings = {
                                "default_voice": voice,
                                "default_speed": speed,
                                "theme": theme,
                                "font_size": font_size,
                                "autoplay": autoplay
                            }
                            
                            if SettingsManager.save_global_settings(settings):
                                return "Settings saved successfully! ✅"
                            else:
                                return "Error saving settings. Check logs for details."
                        except Exception as e:
                            logger.error(f"Error saving global settings: {e}")
                            return f"Error saving settings: {str(e)}"
                    
                    save_globals_btn.click(
                        fn=save_global_settings,
                        inputs=[global_voice, global_speed, theme_selector, font_size, autoplay],
                        outputs=[global_status]
                    )
                
                # Model Management Section - Enhanced
                with gr.Tab("Model Management"):
                    gr.Markdown("## Model Management")
                    
                    with gr.Tabs() as model_tabs:
                        # Task-specific model selection tab
                        with gr.Tab("Task Models"):
                            with gr.Row():
                                with gr.Column():
                                    # Get current task model settings
                                    task_models = SettingsManager.get_task_models()
                                    
                                    # Text generation model
                                    text_model = gr.Dropdown(
                                        choices=initial_models,
                                        label="Text Generation Model",
                                        value=task_models.get("text_model", ""),
                                        info="Used for reading, typing, and math instructions"
                                    )
                                    
                                    # Vision model
                                    vision_model = gr.Dropdown(
                                        choices=initial_multimodal,
                                        label="Vision Model",
                                        value=task_models.get("vision_model", ""),
                                        info="Used for image analysis and document processing"
                                    )
                                    
                                    # Embedding model
                                    embedding_model = gr.Dropdown(
                                        choices=initial_embeddings,
                                        label="Embedding Model",
                                        value=task_models.get("embedding_model", ""),
                                        info="Used for text similarity"
                                    )
                                    
                                    save_task_models_btn = gr.Button("Save Task Models", variant="primary")
                                    task_models_status = gr.Textbox(label="Status")
                        
                        # Model browser tab
                        with gr.Tab("Model Library"):
                            with gr.Row():
                                refresh_models_btn = gr.Button("Refresh Model List 🔄", variant="secondary")
                                library_btn = gr.Button("Browse Ollama Library 🔍", variant="secondary")
                                
                            model_table = gr.Dataframe(
                                headers=["Name", "Size", "Type", "Parameters", "Quantization"],
                                datatype=["str", "str", "str", "str", "str"],
                                value=initial_model_details,
                                row_count=15,
                                interactive=False,
                                col_count=(5, "fixed")
                            )
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    model_name_download = gr.Textbox(
                                        label="Download Model",
                                        placeholder="Enter model name (e.g., llama3, gemma3, or paste 'ollama run model-name')",
                                        elem_classes="model-input"
                                    )
                                
                                with gr.Column(scale=1):
                                    download_btn = gr.Button("Download", variant="primary")
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    model_to_delete = gr.Dropdown(
                                        choices=initial_models,
                                        label="Delete Model",
                                        elem_classes="model-input"
                                    )
                                
                                with gr.Column(scale=1):
                                    delete_btn = gr.Button("Delete", variant="secondary")
                                    
                            library_status = gr.Textbox(label="Status")
                        
                        # Modelfile editor tab - for EDITING existing models
                        with gr.Tab("Model Editor"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("""### Edit Existing Models
                                    
                                    Use this tab to view and modify the modelfile for existing models. 
                                    This lets you adjust parameters like context size, system prompts, and more.
                                    """)
                                    
                                    modelfile_dropdown = gr.Dropdown(
                                        choices=initial_models,
                                        label="Select Model to Edit",
                                        elem_classes="model-selector"
                                    )
                                    
                                    # Using Textbox instead of Code component to avoid language support issues
                                    modelfile_editor = gr.Textbox(
                                        label="Modelfile Content",
                                        lines=20,
                                        elem_classes="modelfile-editor"
                                    )
                                    
                                    with gr.Row():
                                        save_modelfile_btn = gr.Button("Save Modelfile", variant="primary")
                                        
                            with gr.Row():
                                modelfile_status = gr.Textbox(label="Status")
                        
                        # NEW: Create New Model tab - separate from editing
                        with gr.Tab("Create New Model"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("""### Create a New Model
                                    
                                    Create a new model by defining a Modelfile. A Modelfile lets you customize existing models 
                                    with specific parameters, system prompts, etc. 
                                    
                                    **Important**: You must specify:
                                    1. A base model using `FROM` 
                                    2. A new model name in a comment `# NAME: your_model_name`
                                    """)
                                    
                                    # New model name input
                                    new_model_name = gr.Textbox(
                                        label="New Model Name",
                                        placeholder="e.g., my-custom-llama",
                                        elem_classes="new-model-name"
                                    )
                                    
                                    # Base model selector
                                    base_model_dropdown = gr.Dropdown(
                                        choices=initial_models,
                                        label="Base Model",
                                        elem_classes="base-model-selector"
                                    )
                                    
                                    # Template selector
                                    template_dropdown = gr.Dropdown(
                                        choices=["Basic Customization", "Chat Assistant", "RAG Template", "Custom"],
                                        label="Modelfile Template",
                                        value="Basic Customization",
                                        elem_classes="template-selector"
                                    )
                                    
                                    # Modelfile editor for new model
                                    new_modelfile_editor = gr.Textbox(
                                        label="Modelfile Content",
                                        lines=20,
                                        elem_classes="new-modelfile-editor",
                                        placeholder="FROM llama3\n\nSYSTEM \"\"\"You are a helpful assistant.\"\"\""
                                    )
                                    
                                    create_model_btn = gr.Button("Create New Model", variant="primary")
                            
                            with gr.Row():
                                create_model_status = gr.Textbox(label="Status")
                                
                            # Helper function to generate template modelfiles based on selections
                            def update_modelfile_template(name, base, template_type):
                                if not name or not base:
                                    return "# Please fill in the model name and select a base model"
                                
                                # Add the model name as a comment
                                header = f"# NAME: {name}\nFROM {base}\n\n"
                                
                                if template_type == "Basic Customization":
                                    return header + (
                                        "# Basic model customization\n\n"
                                        "# Set parameters\n"
                                        "PARAMETER temperature 0.7\n"
                                        "PARAMETER top_p 0.9\n"
                                        "PARAMETER top_k 40\n\n"
                                        "# Define system message\n"
                                        "SYSTEM \"\"\"You are a helpful, respectful assistant designed to help children learn.\n"
                                        "You explain concepts in simple language appropriate for young learners.\n"
                                        "Your responses are educational, encouraging, and always appropriate for children.\"\"\"\n"
                                    )
                                elif template_type == "Chat Assistant":
                                    return header + (
                                        "# Chat assistant template\n\n"
                                        "# Set parameters for more creative responses\n"
                                        "PARAMETER temperature 0.8\n"
                                        "PARAMETER top_p 0.9\n\n"
                                        "# Define a detailed system message for the assistant\n"
                                        "SYSTEM \"\"\"You are Little Geeky, a friendly AI tutor designed to help children learn.\n"
                                        "- You communicate at a level appropriate for elementary school students\n"
                                        "- You're enthusiastic, supportive, and make learning fun\n"
                                        "- You break down complex concepts into simple explanations\n"
                                        "- You use examples and analogies that children can relate to\n"
                                        "- You're patient and encouraging when students struggle\n"
                                        "- You ask thoughtful questions to guide the learning process\n"
                                        "- You celebrate achievements and progress\n"
                                        "- You never use language that's inappropriate for children\"\"\"\n"
                                    )
                                elif template_type == "RAG Template":
                                    return header + (
                                        "# RAG (Retrieval Augmented Generation) Template\n\n"
                                        "# Set parameters for factual responses\n"
                                        "PARAMETER temperature 0.3\n"
                                        "PARAMETER top_p 0.95\n\n"
                                        "# Define context window for processing\n"
                                        "# You can customize the context window size based on your needs\n"
                                        "# Define the system message for knowledge-based responses\n"
                                        "SYSTEM \"\"\"You are a knowledgeable assistant designed to work with retrieved information.\n"
                                        "When answering questions:\n"
                                        "1. Base your answers primarily on the context information provided\n"
                                        "2. If the context doesn't contain the answer, say you don't know\n"
                                        "3. Don't make up information that isn't in the provided context\n"
                                        "4. Keep your answers concise and focused on the information in the context\n"
                                        "5. Explain concepts in a way that's easy for children to understand\"\"\"\n"
                                    )
                                else:  # "Custom"
                                    return header + (
                                        "# Custom template - add your own customizations\n\n"
                                        "# PARAMETER template 0.7\n"
                                        "# PARAMETER top_p 0.9\n\n"
                                        "# Define your SYSTEM prompt\n"
                                        "SYSTEM \"\"\"Your custom system prompt goes here.\"\"\"\n\n"
                                        "# For more options, see the Ollama documentation:\n"
                                        "# https://github.com/ollama/ollama/blob/main/docs/modelfile.md\n"
                                    )
                            
                            # Wire up the template selector
                            def update_template(name, base, template_type):
                                return update_modelfile_template(name, base, template_type)
                            
                            # Update template when inputs change
                            template_dropdown.change(
                                fn=update_template,
                                inputs=[new_model_name, base_model_dropdown, template_dropdown],
                                outputs=[new_modelfile_editor]
                            )
                            
                            new_model_name.change(
                                fn=update_template,
                                inputs=[new_model_name, base_model_dropdown, template_dropdown],
                                outputs=[new_modelfile_editor]
                            )
                            
                            base_model_dropdown.change(
                                fn=update_template,
                                inputs=[new_model_name, base_model_dropdown, template_dropdown],
                                outputs=[new_modelfile_editor]
                            )
                    
                # Achievement Settings Section
                with gr.Tab("Achievement Settings"):
                    gr.Markdown("## Achievement Settings")
                    
                    # Pre-load achievement data
                    initial_achievements = self._load_achievements()
                    
                    achievements_table = gr.Dataframe(
                        headers=["Achievement", "Description", "Requirements", "Enabled"],
                        datatype=["str", "str", "str", "bool"],
                        value=initial_achievements,
                        row_count=10,
                        interactive=True
                    )
                    
                    with gr.Row():
                        save_achievements_btn = gr.Button("Save Achievement Settings", variant="primary")
                        reset_achievements_btn = gr.Button("Reset to Defaults", variant="secondary")
                        achievements_status = gr.HTML("Ready")
                        
                    def save_achievements(data):
                        """Save achievement configurations"""
                        try:
                            # Map name back to achievement configs and update enabled state
                            for row in data:
                                achievement = next(
                                    (ach for ach in self.app.achievement_manager.achievements.values() 
                                     if ach.name == row[0]), None)
                                if achievement:
                                    achievement.enabled = row[3]
                            
                            self.app.achievement_manager.save_config()
                            return "Achievement settings saved! ✅"
                        except Exception as e:
                            return f"Error saving achievements: {str(e)}"
                        
                    save_achievements_btn.click(
                        fn=save_achievements,
                        inputs=[achievements_table],
                        outputs=[achievements_status]
                    )
                    
                    reset_achievements_btn.click(
                        fn=lambda: (self.app.achievement_manager._create_default_config(), self._load_achievements()),
                        outputs=[achievements_table]
                    )
            
            # Add a prominent refresh button at the bottom of all settings
            with gr.Row():
                refresh_all_btn = gr.Button("Refresh All Settings", variant="primary", scale=2)
                status_all = gr.HTML("Ready")
            
            # === FIXED MODEL MANAGEMENT FUNCTIONS ===
            
            # Function to get models without updating UI components
            async def get_models_only():
                """Get models without updating UI"""
                try:
                    models = await self.app.get_models()
                    return models
                except Exception as e:
                    logger.error(f"Error getting models: {e}")
                    return ["Error getting models"]

            # FIX: Simple refresh handler defined in local scope
            async def handle_refresh():
                """Simple direct refresh function"""
                try:
                    # Get fresh models directly
                    models = await get_models_only()
                    
                    # Create updated model details
                    model_details = []
                    for model_name in models:
                        if model_name not in ["No models found 😕", "Please check if Ollama is running 🤔"]:
                            model_details.append([
                                model_name, 
                                "Ready",
                                "Ready",
                                "Ready", 
                                "Ready"
                            ])
                    
                    # Update filtered lists
                    multimodal_models = self._get_multimodal_models(models)
                    embedding_models = self._get_embedding_models(models)
                    
                    message = f"Models refreshed successfully! Found {len(models)} models."
                    
                    # Return updated values
                    return message, model_details, gr.update(choices=models), gr.update(choices=multimodal_models), gr.update(choices=embedding_models), gr.update(choices=models), gr.update(choices=models), gr.update(choices=models)
                except Exception as e:
                    logger.error(f"Error in refresh handler: {e}")
                    return f"Error refreshing models: {str(e)}", [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            # Simplified download function with preprocessing for model names
            async def download_model(model_name):
                """Download a model without automatically updating UI"""
                if not model_name:
                    return "Please enter a model name to download"
                
                try:
                    # Preprocess model name to strip common prefixes
                    original_name = model_name
                    if "ollama run " in model_name:
                        model_name = model_name.replace("ollama run ", "")
                        logger.info(f"Removed 'ollama run' prefix from model name: {original_name} -> {model_name}")
                    
                    # Use Ollama CLI to download model
                    process = await asyncio.create_subprocess_exec(
                        "ollama", "pull", model_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        if original_name != model_name:
                            return f"Processed '{original_name}' as '{model_name}' and downloaded successfully! Use Refresh to update the model list."
                        else:
                            return f"Model {model_name} downloaded successfully! Use Refresh to update the model list."
                    else:
                        return f"Failed to download model: {stderr.decode()}"
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    return f"Error downloading model: {str(e)}"
            
            # Simplified delete function
            async def delete_model(model_name):
                """Delete a model without automatically updating UI"""
                if not model_name:
                    return "Please select a model to delete"
                
                try:
                    # Use Ollama API to delete model
                    async with aiohttp.ClientSession() as session:
                        async with session.delete(
                            f"{OLLAMA_API_URL}/delete",
                            json={"model": model_name}
                        ) as response:
                            if response.status == 200:
                                return f"Model {model_name} deleted successfully! Use Refresh to update the model list."
                            else:
                                error_text = await response.text()
                                return f"Failed to delete model: {error_text}"
                except Exception as e:
                    logger.error(f"Error deleting model: {e}")
                    return f"Error deleting model: {str(e)}"
            
            # Load modelfile content
            async def load_modelfile(model_name):
                """Load a modelfile without updating other components"""
                if not model_name:
                    return "", "Please select a model"
                
                try:
                    # Use Ollama API to get model info
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{OLLAMA_API_URL}/show",
                            json={"model": model_name}
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                modelfile_content = data.get("modelfile", "# No modelfile content available")
                                return modelfile_content, "Modelfile loaded successfully"
                            else:
                                error_text = await response.text()
                                return "", f"Failed to load modelfile: {error_text}"
                except Exception as e:
                    logger.error(f"Error loading modelfile: {e}")
                    return "", f"Error loading modelfile: {str(e)}"
            
            # Save modelfile
            async def save_modelfile(model_name, modelfile_content):
                """Save modelfile without updating UI components"""
                if not model_name or not modelfile_content:
                    return "Model name and modelfile content are required"
                
                try:
                    # Write modelfile to temp file
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    modelfile_path = os.path.join(temp_dir, "Modelfile")
                    
                    with open(modelfile_path, "w") as f:
                        f.write(modelfile_content)
                    
                    # Use Ollama CLI to update model
                    process = await asyncio.create_subprocess_exec(
                        "ollama", "create", model_name, "-f", modelfile_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    # Clean up temp file
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                    if process.returncode == 0:
                        return f"Model {model_name} updated successfully!"
                    else:
                        return f"Error updating model: {stderr.decode()}"
                except Exception as e:
                    logger.error(f"Error saving modelfile: {e}")
                    return f"Error saving modelfile: {str(e)}"
            
            # Create new model
            async def create_new_model(model_name, modelfile_content):
                """Create a new model without updating UI components"""
                if not model_name or not modelfile_content:
                    return "Model name and modelfile content are required"
                
                try:
                    # Write modelfile to temp file
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    modelfile_path = os.path.join(temp_dir, "Modelfile")
                    
                    with open(modelfile_path, "w") as f:
                        f.write(modelfile_content)
                    
                    # Use Ollama CLI to create model
                    process = await asyncio.create_subprocess_exec(
                        "ollama", "create", model_name, "-f", modelfile_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    # Clean up temp file
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                    if process.returncode == 0:
                        return f"Model {model_name} created successfully! Use Refresh to update the model list."
                    else:
                        return f"Error creating model: {stderr.decode()}"
                except Exception as e:
                    logger.error(f"Error creating model: {e}")
                    return f"Error creating model: {str(e)}"
            
            # Save task models
            async def save_task_models(text_model, vision_model, embedding_model):
                """Save task models configuration"""
                try:
                    settings = {
                        "text_model": text_model,
                        "vision_model": vision_model,
                        "embedding_model": embedding_model
                    }
                    
                    if SettingsManager.save_task_models(settings):
                        return "Task models saved successfully!"
                    else:
                        return "Error saving task models. Check logs for details."
                except Exception as e:
                    logger.error(f"Error saving task models: {e}")
                    return f"Error saving task models: {str(e)}"
            
            # Browser launch function
            def open_ollama_library():
                """Open Ollama library in browser"""
                try:
                    webbrowser.open("https://ollama.com/library")
                    return "Opened Ollama model library in browser"
                except Exception as e:
                    logger.error(f"Error opening browser: {e}")
                    return f"Error opening browser: {str(e)}"
            
            # Refresh all settings - FIXED VERSION
            async def refresh_all_settings():
                """Refresh all components"""
                try:
                    # Update model components using the handle_refresh function
                    refresh_result = await handle_refresh()
                    
                    # Apply the model updates with proper handling for update objects
                    library_status.update(value=refresh_result[0])
                    model_table.update(value=refresh_result[1])
                    
                    # The issue is in these lines - we need to handle different Gradio versions
                    # Check if we have gr.update objects or direct choice lists
                    def safe_update_dropdown(dropdown, choices_data):
                        try:
                            # Direct update approach for Gradio 3.41.0
                            if isinstance(choices_data, dict) and 'choices' in choices_data:
                                # Handle gr.update() return value
                                dropdown.update(choices=choices_data['choices'])
                            elif hasattr(choices_data, 'choices'):
                                # Handle objects with choices attribute  
                                dropdown.update(choices=choices_data.choices)
                            else:
                                # Fall back to direct update for lists or other formats
                                dropdown.update(choices=choices_data)
                        except Exception as e:
                            # Log the specific error for debugging
                            logger.error(f"Error updating dropdown: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
                    # Update all dropdowns safely
                    try:
                        safe_update_dropdown(text_model, refresh_result[2])
                    except Exception as e:
                        logger.error(f"Error updating text_model: {e}")
                        
                    try:
                        safe_update_dropdown(vision_model, refresh_result[3])
                    except Exception as e:
                        logger.error(f"Error updating vision_model: {e}")
                        
                    try:
                        safe_update_dropdown(embedding_model, refresh_result[4])
                    except Exception as e:
                        logger.error(f"Error updating embedding_model: {e}")
                        
                    try:
                        safe_update_dropdown(model_to_delete, refresh_result[5])
                    except Exception as e:
                        logger.error(f"Error updating model_to_delete: {e}")
                        
                    try:
                        safe_update_dropdown(modelfile_dropdown, refresh_result[6])
                    except Exception as e:
                        logger.error(f"Error updating modelfile_dropdown: {e}")
                        
                    try:
                        safe_update_dropdown(base_model_dropdown, refresh_result[7])
                    except Exception as e:
                        logger.error(f"Error updating base_model_dropdown: {e}")
                    
                    # Refresh achievements table
                    achievements_table.update(value=self._load_achievements())
                    
                    return "<span style='color:green'>All settings refreshed successfully!</span>"
                except Exception as e:
                    logger.error(f"Error refreshing all settings: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return f"<span style='color:red'>Error refreshing settings: {str(e)}</span>"
            
            # FIX: Wire up refresh_models_btn with local handler function
            refresh_models_btn.click(
                fn=handle_refresh,
                outputs=[
                    library_status, 
                    model_table, 
                    text_model, 
                    vision_model, 
                    embedding_model, 
                    model_to_delete, 
                    modelfile_dropdown, 
                    base_model_dropdown
                ]
            )
            
            library_btn.click(
                fn=open_ollama_library,
                outputs=[library_status]
            )
            
            download_btn.click(
                fn=download_model,
                inputs=[model_name_download],
                outputs=[library_status]
            )
            
            delete_btn.click(
                fn=delete_model,
                inputs=[model_to_delete],
                outputs=[library_status]
            )
            
            modelfile_dropdown.change(
                fn=load_modelfile,
                inputs=[modelfile_dropdown],
                outputs=[modelfile_editor, modelfile_status]
            )
            
            save_modelfile_btn.click(
                fn=save_modelfile,
                inputs=[modelfile_dropdown, modelfile_editor],
                outputs=[modelfile_status]
            )
            
            create_model_btn.click(
                fn=create_new_model,
                inputs=[new_model_name, new_modelfile_editor],
                outputs=[create_model_status]
            )
            
            save_task_models_btn.click(
                fn=save_task_models,
                inputs=[text_model, vision_model, embedding_model],
                outputs=[task_models_status]
            )
            
            refresh_all_btn.click(
                fn=refresh_all_settings,
                outputs=[status_all]
            )
            
            return tab
            
    def _load_achievements(self):
        """Load achievement configurations"""
        try:
            achievements = []
            for a in self.app.achievement_manager.achievements.values():
                requirements = ", ".join(f"{k}: {v}" for k, v in a.requirements.items())
                achievements.append([a.name, a.description, requirements, a.enabled])
            return achievements
        except Exception as e:
            logger.error(f"Error loading achievements: {e}")
            return []
    
    def _generate_preview_html(self, theme, font_size):
        """Generate HTML for theme preview"""
        bg_color = {
            "Blue Theme": "#295095",
            "Dark Theme": "#222222",
            "Light Theme": "#f0f0f0",
            "High Contrast": "#000000"
        }.get(theme, "#295095")
        
        text_color = "#ffffff" if theme in ["Blue Theme", "Dark Theme", "High Contrast"] else "#000000"
        
        return f"""
        <div style="border: 1px solid #ccc; padding: 15px; border-radius: 8px;">
            <h3 style="font-size: {1.2 * font_size}em;">Settings Preview</h3>
            <p style="font-size: {font_size}em;">This is how your text will appear.</p>
            <div style="margin-top: 10px;">
                <div style="background: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; font-size: {font_size}em;">
                    {theme} Preview
                </div>
            </div>
        </div>
        """
            
    def _get_model_names(self):
        """Get a list of model names for dropdowns"""
        try:
            # Get any cached models from Ollama
            import aiohttp
            import asyncio
            
            async def fetch_models():
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{OLLAMA_API_URL}/tags") as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'models' in data and data['models']:
                                return [model['name'] for model in data['models']]
                        return []
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            models = loop.run_until_complete(fetch_models())
            
            if not models:
                return ["No models found 😕"]
                
            return models
        except Exception as e:
            logger.error(f"Error getting model names: {e}")
            return ["Error loading models"]
    
    def _get_multimodal_models(self, models=None):
        """Get a list of multimodal models that can process images"""
        if not models:
            models = self._get_model_names()
            
        # Filter for known multimodal models
        multimodal_keywords = ["llava", "vision", "bakllava", "cogvlm", "multimodal", "clip"]
        multimodal_models = [
            model for model in models 
            if any(keyword in model.lower() for keyword in multimodal_keywords)
        ]
        
        if not multimodal_models:
            # Add instruction to download if none found
            multimodal_models = ["No vision models found - download llava from library"]
            
        return multimodal_models
    
    def _get_embedding_models(self, models=None):
        """Get a list of embedding models"""
        if not models:
            models = self._get_model_names()
            
        # Filter for known embedding models
        embedding_keywords = ["embed", "minilm", "sentence", "bert", "clip"]
        embedding_models = [
            model for model in models 
            if any(keyword in model.lower() for keyword in embedding_keywords)
        ]
        
        if not embedding_models:
            # Add instruction to download if none found
            embedding_models = ["No embedding models found - download all-minilm from library"]
            
        return embedding_models
    
    def _get_model_details_sync(self, models):
        """Synchronous version of _get_model_details for initial loading"""
        try:
            # Create a simplified model details table for initial load
            model_details = []
            for model_name in models:
                if model_name in ["No models found 😕", "Please check if Ollama is running 🤔"]:
                    continue
                
                model_details.append([
                    model_name,
                    "Ready",  # Changed from "Loading..." to "Ready"
                    "Ready",  # Changed from "Loading..." to "Ready"
                    "Ready",  # Changed from "Loading..." to "Ready"
                    "Ready"   # Changed from "Loading..." to "Ready"
                ])
            return model_details
        except Exception as e:
            logger.error(f"Error getting initial model details: {e}")
            return []
    
    async def _get_model_details(self, models):
        """Get detailed information about models"""
        try:
            model_details = []
            
            async with aiohttp.ClientSession() as session:
                for model_name in models:
                    try:
                        if model_name in ["No models found 😕", "Please check if Ollama is running 🤔"]:
                            continue
                            
                        async with session.post(
                            f"{OLLAMA_API_URL}/show",
                            json={"model": model_name}
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                details = data.get("details", {})
                                
                                size = "Unknown"
                                model_type = details.get("format", "Unknown")
                                params = details.get("parameter_size", "Unknown")
                                quant = details.get("quantization_level", "None")
                                
                                model_details.append([
                                    model_name, 
                                    size,
                                    model_type,
                                    params,
                                    quant
                                ])
                    except Exception as e:
                        logger.error(f"Error getting details for model {model_name}: {e}")
                        model_details.append([model_name, "Error", "Error", "Error", "Error"])
                        
            return model_details
        except Exception as e:
            logger.error(f"Error getting model details: {e}")
            return []
