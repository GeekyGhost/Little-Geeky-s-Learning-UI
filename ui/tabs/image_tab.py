# ui/tabs/image_tab.py
import gradio as gr
import os
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import rembg  # Optional, for background removal

from utils.logging_utils import logger
from models.user_progress import UserProgress
from utils.settings_utils import SettingsManager
from services.image_service import StableDiffusionProcessor

class ImageTab:
    def __init__(self, app_context):
        self.app = app_context
        self.image_processor = StableDiffusionProcessor()
        self.last_generated_images = []
        self.last_seed = -1
        
    def create_tab(self) -> gr.Tab:
        # Get global voice settings for consistency
        voice_settings = SettingsManager.get_voice_settings()
        default_speed = voice_settings.get("speed", 1.0)
        
        with gr.Tab("Image Creator 🎨") as tab:
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        lines=3, 
                        label="Describe the image you want",
                        placeholder="A cute cartoon character with a round face and big eyes",
                        elem_classes="image-prompt"
                    )
                    
                    negative_prompt = gr.Textbox(
                        lines=2,
                        label="What to avoid in the image",
                        placeholder="blurry, bad quality, disfigured, ugly",
                        elem_classes="negative-prompt"
                    )
                    
                    with gr.Row():
                        model_type = gr.Radio(
                            choices=["LCM (Fast)", "Standard"],
                            value="LCM (Fast)",
                            label="Generation Speed",
                            elem_classes="model-selector"
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            steps = gr.Slider(
                                minimum=1, 
                                maximum=50, 
                                value=4, 
                                step=1, 
                                label="Steps"
                            )
                            
                            cfg_scale = gr.Slider(
                                minimum=1.0, 
                                maximum=15.0, 
                                value=1.5, 
                                step=0.1, 
                                label="Guidance Scale"
                            )
                            
                        with gr.Column():
                            width = gr.Slider(
                                minimum=256, 
                                maximum=1024, 
                                value=512, 
                                step=64, 
                                label="Width"
                            )
                            
                            height = gr.Slider(
                                minimum=256, 
                                maximum=1024, 
                                value=512, 
                                step=64, 
                                label="Height"
                            )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Number(
                            value=-1, 
                            label="Seed (-1 for random)",
                            precision=0
                        )
                        
                        transparent_bg = gr.Checkbox(
                            label="Transparent Background", 
                            value=False
                        )
                        
                        sampler = gr.Dropdown(
                            choices=["LCM", "Euler a", "DPM++ SDE", "DDIM"],
                            value="LCM",
                            label="Sampler"
                        )
                        
                        scheduler = gr.Dropdown(
                            choices=["sgm_uniform", "karras", "normal", "simple"],
                            value="sgm_uniform",
                            label="Scheduler"
                        )
                    
                    style_templates = gr.Dropdown(
                        choices=["None", "Cartoon", "Watercolor", "3D Render", "Pixel Art"],
                        value="None",
                        label="Style Templates"
                    )
                    
                    with gr.Row():
                        gen_btn = gr.Button("Generate Image 🎨", variant="primary")
                        save_btn = gr.Button("Save to Assets 💾")
                
                with gr.Column(scale=1):
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        stats_val = progress.get_stats_summary()
                    else:
                        stats_val = {"Total Images": 0, "Last Active": None}
                    
                    stats = gr.JSON(
                        value=stats_val,
                        label="Your Progress",
                        every=1
                    )
                    
                    output_image = gr.Image(
                        type="pil",
                        label="Generated Image",
                        elem_classes="output-image"
                    )
                    
                    image_info = gr.JSON(
                        value={},
                        label="Image Details"
                    )
            
            # Define UI update functions based on template selection
            def apply_template(template_name):
                if template_name == "None":
                    return "", ""
                elif template_name == "Cartoon":
                    return "a cute cartoon character, bright colors, simple shapes, child-friendly", "realistic, detailed, photorealistic, 3d render, photograph"
                elif template_name == "Watercolor":
                    return "watercolor painting style, soft colors, artistic, flowing colors", "digital art, 3d, sharp lines, photorealistic"
                elif template_name == "3D Render":
                    return "3D render, CGI, smooth textures, digital art", "sketch, drawing, 2d, flat"
                elif template_name == "Pixel Art":
                    return "pixel art style, 8-bit graphics, retro game style", "realistic, 3d, detailed, high resolution"
            
            # Update UI based on model type
            def update_model_settings(model_type):
                if model_type == "LCM (Fast)":
                    return gr.update(value=4, maximum=8), gr.update(value=1.5, maximum=3.0), gr.update(value="LCM"), gr.update(value="sgm_uniform")
                else:
                    return gr.update(value=30, maximum=50), gr.update(value=7.5, maximum=15.0), gr.update(value="Euler a"), gr.update(value="karras")
            
            # Main generation function
            async def generate_image(prompt, negative_prompt, model_type, steps, cfg_scale, 
                                    width, height, seed, transparent_bg, sampler, scheduler):
                try:
                    # Initialize stats_data at the beginning of the function
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        stats_data = progress.get_stats_summary()
                    else:
                        stats_data = {"Total Images": 0, "Last Active": None}
                    
                    # Seed handling - either use provided seed or generate one
                    if seed == -1:
                        import random
                        seed = random.randint(0, 2147483647)
                        logger.info(f"Generated random seed: {seed}")
                    else:
                        logger.info(f"Using fixed seed: {seed}")
                    
                    # Call the image processor service
                    image, image_metadata = await self.image_processor.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model_type=model_type,
                        steps=steps,
                        guidance_scale=cfg_scale,
                        width=width,
                        height=height,
                        seed=seed,
                        sampler=sampler,
                        scheduler=scheduler
                    )
                    
                    # Apply background removal if requested
                    if transparent_bg and image:
                        image = rembg.remove(image)
                    
                    # Update stats if user is logged in
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        progress.update_stat("images_generated", 1)
                        await self.app.achievement_manager.check_achievements(progress)
                        stats_data = progress.get_stats_summary()
                    
                    # Save the last generated image and seed
                    self.last_generated_images = [image]
                    self.last_seed = seed
                    
                    # Make sure metadata includes the seed
                    if isinstance(image_metadata, dict) and "seed" not in image_metadata:
                        image_metadata["seed"] = seed
                    
                    return image, image_metadata, stats_data
                except Exception as e:
                    logger.error(f"Error generating image: {e}")
                    return None, {"error": str(e), "seed": seed}, stats_data
            
            # Save image to assets
            def save_to_assets(image):
                if image is None:
                    return "No image to save!"
                
                try:
                    from config.settings import ASSETS_IMAGES_DIR  # Import from settings
                    
                    # Generate filename with timestamp
                    import time
                    filename = f"image_{int(time.time())}.png"
                    filepath = ASSETS_IMAGES_DIR / filename
                    
                    # Save image
                    image.save(filepath)
                    
                    return f"Image saved to assets: {filename}"
                except Exception as e:
                    logger.error(f"Error saving image: {e}")
                    return f"Error saving image: {str(e)}"
            
            # Connect UI components to functions
            style_templates.change(
                fn=apply_template,
                inputs=[style_templates],
                outputs=[prompt, negative_prompt]
            )
            
            model_type.change(
                fn=update_model_settings,
                inputs=[model_type],
                outputs=[steps, cfg_scale, sampler, scheduler]
            )
            
            gen_btn.click(
                fn=generate_image,
                inputs=[
                    prompt, negative_prompt, model_type, steps, cfg_scale,
                    width, height, seed, transparent_bg, sampler, scheduler
                ],
                outputs=[output_image, image_info, stats]
            )
            
            save_btn.click(
                fn=save_to_assets,
                inputs=[output_image],
                outputs=[image_info]
            )
            
            return tab