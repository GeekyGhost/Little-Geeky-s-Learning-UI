# services/image_service.py - Complete Offline Fix
import os
import torch
import random
import numpy as np
import gc  # For garbage collection
from PIL import Image
from pathlib import Path
from utils.logging_utils import logger
from config.settings import BASE_DIR

# Suppress common warnings
import warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="The class CLIPFeatureExtractor is deprecated")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

# Force offline mode via environment variables - MUST be set before importing diffusers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Monkey patch urllib3 to block any GitHub requests
def patch_urllib3():
    """Patch urllib3 to block any requests to GitHub"""
    try:
        import urllib3.connectionpool
        from urllib3.exceptions import MaxRetryError
        
        # Store the original function
        original_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen
        
        # Define a patched version
        def patched_urlopen(self, method, url, **kw):
            # Block GitHub requests completely
            if "githubusercontent.com" in self.host:
                raise MaxRetryError(self, url, reason=f"Offline mode: blocked request to {self.host}")
            return original_urlopen(self, method, url, **kw)
        
        # Apply the patch
        urllib3.connectionpool.HTTPConnectionPool.urlopen = patched_urlopen
        logger.info("Patched urllib3 to block GitHub requests")
    except Exception as e:
        logger.warning(f"Could not patch urllib3: {e}")

# Apply the patch immediately
patch_urllib3()

def ensure_config_file_exists(model_path):
    """Create a config file next to the model file"""
    # Get the directory where the model is stored
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, "v1-inference.yaml")
    
    # If config file already exists, use it
    if os.path.exists(config_path):
        logger.info(f"Using existing config file: {config_path}")
        return config_path
    
    # Create a default config file
    logger.info("Creating default config file...")
    default_config = """model:
  base_learning_rate: 1.0e-4
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: true
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder"""
    
    with open(config_path, "w") as f:
        f.write(default_config)
    logger.info(f"Created default config file at: {config_path}")
    return config_path

class StableDiffusionProcessor:
    def __init__(self):
        self.models = {}
        self.checkpoints_dir = Path("Checkpoints")

        # FIXED: Simplified but robust GPU detection
        self.device = "cpu"
        if torch.cuda.is_available():
            # Enable TF32 for better performance on RTX cards
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Using GPU: {device_name} with {device_mem:.2f} GB memory")
        else:
            logger.warning("CUDA not available. Using CPU (will be very slow)")
        
        # Create assets directory
        self.assets_dir = BASE_DIR / "assets" / "images"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_image(self, prompt, negative_prompt, model_type, steps, 
                           guidance_scale, width, height, seed, sampler, scheduler):
        """Generate an image using Stable Diffusion with better GPU handling"""
        # Force dimensions to be multiples of 8 (SD requirement)
        width = (width // 8) * 8
        height = (height // 8) * 8
        
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
                # Force optimal LCM parameters
                if steps > 50:
                    steps = min(steps, 50)
                if guidance_scale > 2.0:
                    guidance_scale = min(guidance_scale, 2.0)
                
                # Find model and load pipeline if not already loaded
                if "lcm" not in self.models:
                    # Look for LCM models with various naming patterns
                    model_files = list(self.checkpoints_dir.glob("*LCM*.safetensors"))
                    
                    if not model_files:
                        # Also look for any safetensors file if no LCM specifically named ones exist
                        model_files = list(self.checkpoints_dir.glob("*.safetensors"))
                    
                    if not model_files:
                        error_msg = (
                            "No models found in Checkpoints folder. Please download a Stable Diffusion "
                            "model and place it in the Checkpoints directory."
                        )
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                    
                    # Use the first found model file
                    model_path = str(model_files[0])
                    logger.info(f"Loading model from {model_path}")
                    
                    # Ensure the config file exists alongside the model
                    config_file = ensure_config_file_exists(model_path)
                    
                    # FIXED: Proper memory cleanup before loading model
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # FIXED: Ensure we use float16 for GPU, which is critical for stable diffusion
                    dtype = torch.float16 if self.device == "cuda" else torch.float32
                    
                    # FIXED: Specify the config file explicitly and use local_files_only=True
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_path,
                        original_config_file=config_file,  # Specify the local config file
                        local_files_only=True,              # Force offline mode
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    # FIXED: Explicit pipeline device placement
                    pipe = pipe.to(self.device)
                    
                    # Set the correct scheduler for LCM
                    if sampler.lower() == "lcm":
                        pipe.scheduler = LCMScheduler.from_config(
                            pipe.scheduler.config,
                            timestep_spacing="leading",
                            beta_schedule="scaled_linear"
                        )
                    elif sampler.lower() == "euler a":
                        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    elif sampler.lower() == "dpm++ sde":
                        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                    elif sampler.lower() == "ddim":
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                    # Memory optimizations that work well in other UIs
                    pipe.enable_attention_slicing(1)
                    pipe.enable_vae_slicing()
                    
                    # Store the pipeline
                    self.models["lcm"] = pipe
                    logger.info("LCM model loaded and optimized")
                
                # Get the cached pipeline
                pipe = self.models["lcm"]
            else:
                # Standard model setup - similar approach as LCM
                if "standard" not in self.models:
                    model_files = list(self.checkpoints_dir.glob("*.safetensors"))
                    
                    if not model_files:
                        error_msg = "No models found in the Checkpoints directory."
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                    
                    model_path = str(model_files[0])
                    logger.info(f"Loading standard model from {model_path}")
                    
                    # Ensure the config file exists alongside the model
                    config_file = ensure_config_file_exists(model_path)
                    
                    # Free up memory
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Set precision
                    dtype = torch.float16 if self.device == "cuda" else torch.float32
                    
                    # FIXED: Specify the config file explicitly and use local_files_only=True
                    pipe = StableDiffusionPipeline.from_single_file(
                        model_path,
                        original_config_file=config_file,  # Specify the local config file
                        local_files_only=True,              # Force offline mode
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    # Explicit device placement
                    pipe = pipe.to(self.device)
                    
                    # Configure scheduler
                    if sampler.lower() == "euler a":
                        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    elif sampler.lower() == "dpm++ sde":
                        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
                    elif sampler.lower() == "ddim":
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                    # Memory optimizations
                    pipe.enable_attention_slicing(1)
                    pipe.enable_vae_slicing()
                    
                    # Store the pipeline
                    self.models["standard"] = pipe
                    logger.info("Standard model loaded and optimized")
                
                pipe = self.models["standard"]
            
            # Create deterministic generator with the provided seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
                logger.info(f"Generated random seed: {seed}")
            else:
                logger.info(f"Using fixed seed: {seed}")
            
            # FIXED: Create generator on the correct device explicitly
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Clear memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log device info right before generation for diagnostics
            logger.info(f"Starting image generation on {self.device}")
            
            # Generate image with torch inference mode for better memory handling
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            logger.info(f"Image generation completed successfully on {self.device}")
            
            # Clean up memory after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create metadata
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
                "scheduler": scheduler,
                "device": self.device
            }
            
            return result.images[0], image_metadata
                
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            # Clean up memory after error
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            raise e
    
    def unload_models(self):
        """Explicitly unload all models to free memory"""
        self.models = {}
        
        # Force garbage collection
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("All models unloaded from memory")