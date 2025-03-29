# services/image_service.py
import os
import torch
import random
import numpy as np
import gc  # Add import for garbage collection
from PIL import Image
from pathlib import Path
from utils.logging_utils import logger
from config.settings import BASE_DIR

# Add these at the top to suppress warnings
import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="The class CLIPFeatureExtractor is deprecated")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

class StableDiffusionProcessor:
    def __init__(self):
        self.models = {}
        self.checkpoints_dir = Path("Checkpoints")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add optimization settings
        self.attention_slicing = True
        self.enable_vae_slicing = True
        self.enable_xformers = torch.cuda.is_available()
        self.torch_compile = False  # Set to True if using PyTorch 2.0+
        
        # Create assets directory structure
        self.assets_dir = BASE_DIR / "assets" / "images"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created assets directory at {self.assets_dir}")
        
    async def generate_image(self, prompt, negative_prompt, model_type, steps, 
                           guidance_scale, width, height, seed, sampler, scheduler):
        """Generate an image using Stable Diffusion with optimized parameters"""
        try:
            # Import libraries here to avoid loading them at startup
            from diffusers import (
                StableDiffusionPipeline,
                LCMScheduler,
                DDIMScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverSinglestepScheduler
            )
            from diffusers.utils import logging as diffusers_logging
            
            # Set diffusers verbosity to error only
            diffusers_logging.set_verbosity_error()
            
            # Optimize parameters based on model type
            if model_type == "LCM (Fast)":
                # Optimize steps and guidance for LCM
                if steps > 8:
                    logger.info(f"Reducing steps for LCM from {steps} to 8")
                    steps = 8
                if guidance_scale > 2.0:
                    logger.info(f"Reducing guidance scale for LCM from {guidance_scale} to 2.0")
                    guidance_scale = 2.0
                
                # Check if we have the model loaded already
                if "lcm" not in self.models:
                    # Look for LCM model in Checkpoints directory
                    model_files = list(self.checkpoints_dir.glob("*LCM*.safetensors"))
                    
                    if not model_files:
                        # Also check for any .safetensors file if no LCM specifically named ones exist
                        model_files = list(self.checkpoints_dir.glob("*.safetensors"))
                    
                    if not model_files:
                        error_msg = (
                            "No LCM model found in Checkpoints folder. Please download a model "
                            "and place it in the Checkpoints directory with 'LCM' in the filename."
                        )
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                    
                    # Use the first found model file
                    model_path = str(model_files[0])
                    logger.info(f"Loading local LCM model from {model_path}")
                    
                    # Create a pipeline with optimized settings
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    # Configure LCM scheduler based on selection
                    if sampler == "LCM":
                        try:
                            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                        except:
                            logger.warning("LCM scheduler not available, falling back to DPM solver")
                            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "Euler a":
                        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "DPM++ SDE":
                        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "DDIM":
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                    # Apply memory optimizations
                    if self.attention_slicing:
                        pipe.enable_attention_slicing()
                    if self.enable_vae_slicing:
                        pipe.enable_vae_slicing()
                    if self.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                        pipe.enable_xformers_memory_efficient_attention()
                    
                    # Move to device
                    pipe = pipe.to(self.device)
                    
                    # Store model for reuse
                    self.models["lcm"] = pipe
                    logger.info(f"LCM model loaded and optimized")
                
                pipe = self.models["lcm"]
                
            else:  # Standard model
                if "standard" not in self.models:
                    # Look for standard model in Checkpoints directory
                    model_files = list(self.checkpoints_dir.glob("*.safetensors"))
                    
                    if not model_files:
                        error_msg = (
                            "No model found in Checkpoints folder. Please download a Stable Diffusion "
                            "model and place it in the Checkpoints directory."
                        )
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                    
                    # Use the first found model file
                    model_path = str(model_files[0])
                    logger.info(f"Loading local standard model from {model_path}")
                    
                    # Create a pipeline with optimized settings
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    # Configure scheduler based on selection
                    if sampler == "Euler a":
                        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "DPM++ SDE":
                        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                    elif sampler == "DDIM":
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                    # Apply memory optimizations
                    if self.attention_slicing:
                        pipe.enable_attention_slicing()
                    if self.enable_vae_slicing:
                        pipe.enable_vae_slicing()
                    if self.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                        pipe.enable_xformers_memory_efficient_attention()
                    
                    # Move to device
                    pipe = pipe.to(self.device)
                    
                    # Store model for reuse
                    self.models["standard"] = pipe
                    logger.info(f"Standard model loaded and optimized")
                
                pipe = self.models["standard"]
            
            # Create deterministic generator with the provided seed
            if seed == -1:
                # Generate a random seed if -1 is provided
                seed = random.randint(0, 2147483647)
                logger.info(f"Generated random seed: {seed}")
            else:
                logger.info(f"Using fixed seed: {seed}")
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
            # Clean up memory after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create metadata for the image
            image_metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "model": model_type,
                "sampler": sampler,
                "scheduler": scheduler
            }
            
            return result.images[0], image_metadata
                
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            # Clean up memory after error
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            raise e