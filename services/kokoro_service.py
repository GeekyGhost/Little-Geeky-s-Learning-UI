# services/kokoro_service.py
import os
import time
import torch
import numpy as np
import tempfile
import threading
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from utils.logging_utils import logger
import gc  # For garbage collection
import soundfile as sf

# Import our optimized phoneme processor
from services.phoneme_service import FastPhonemeProcessor, PhonemeCache

# Try to import kokoro components with error handling
try:
    from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
    logger.info("Kokoro TTS is available!")
except ImportError as e:
    KOKORO_AVAILABLE = False
    logger.error(f"Kokoro import error: {e}. Using fallback TTS system.")
    logger.info("To install Kokoro, run: pip install kokoro")

# Try to import Chinese normalization (safely)
try:
    from char_convert import tranditional_to_simplified
    CHINESE_SUPPORT = True
except ImportError:
    CHINESE_SUPPORT = False

@dataclass
class KokoroVoiceConfig:
    """Voice configuration for Kokoro TTS"""
    name: str
    id: str
    gender: str
    language: str  # 'us' or 'uk'
    description: Optional[str] = None

class KokoroTTS:
    """
    Handles text-to-speech processing using Kokoro with optimized phonemization
    """
    # Class variables to maintain state across instances
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    VOICE_DATA_CACHE = {}  # Voice data cache
    MODEL_LOCK = threading.Lock()
    
    # Maximum phoneme length allowed by the model
    MAX_PHONEME_LENGTH = 500
    
    # Increased target length for better chunking
    TARGET_LENGTH = 400  # Increased from 150 for better performance
    
    def __init__(self):
        """Initialize the Kokoro TTS system"""
        self._initialize()
        
    def _initialize(self):
        """Initialize with lazy loading approach"""
        if not KOKORO_AVAILABLE:
            logger.warning("Kokoro package not available. Will use fallback TTS.")
            return
        
        logger.info("Initializing Kokoro TTS with phoneme optimization...")
        
        try:
            # Define voice configurations
            self._define_all_voices()
            
            # Initialize pipelines for US and UK English
            for code in ['a', 'b']:
                if code not in self.PIPELINES:
                    self.PIPELINES[code] = KPipeline(lang_code=code, model=False)
            
            # Initialize model only if needed
            with self.MODEL_LOCK:
                if self.MODEL is None:
                    self.MODEL = {}
                    
                    # Initialize model with eval mode for inference
                    model = KModel().to('cpu').eval()
                    
                    # Use torch.jit.script for optimization if possible
                    try:
                        # Attempt to script the model for better performance
                        self.MODEL[False] = torch.jit.script(model)
                        logger.info("Using TorchScript optimization for CPU")
                    except Exception as e:
                        logger.warning(f"TorchScript optimization failed: {e}, using standard model")
                        self.MODEL[False] = model
                    
                    if torch.cuda.is_available():
                        logger.info("CUDA available. GPU model will be loaded on demand.")
                        
                        # Set CUDA optimization flags
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
            
            # Preload default voice if specified in settings
            try:
                self._preload_default_voice()
            except Exception as e:
                logger.warning(f"Could not preload default voice: {e}")
                
            # Initialize phoneme cache
            PhonemeCache.initialize()
                
            logger.info(f"Kokoro TTS initialization complete with {len(self.VOICES)} voices.")
        except Exception as e:
            logger.error(f"Kokoro initialization error: {e}")
            self.VOICES = {}
    
    def _preload_default_voice(self):
        """Preload the default voice from settings to improve first-call performance"""
        try:
            from utils.settings_utils import SettingsManager
            
            # Get voice settings
            voice_settings = SettingsManager.get_voice_settings()
            default_voice = voice_settings.get("voice", "System Default")
            
            # Check if this is a Kokoro voice
            if "Kokoro" in default_voice and ":" in default_voice:
                # Extract name
                voice_name = default_voice.split(": ")[1]
                if voice_name in self.VOICES:
                    voice_config = self.VOICES[voice_name]
                    lang_code = 'a' if voice_config.language == 'us' else 'b'
                    pipeline = self.PIPELINES[lang_code]
                    
                    # Preload voice data into cache
                    self.VOICE_DATA_CACHE[voice_name] = pipeline.load_voice(voice_config.id)
                    logger.info(f"Preloaded default voice: {voice_name}")
        except Exception as e:
            logger.warning(f"Error preloading default voice: {e}")
            
    def _define_all_voices(self):
        """Define all possible Kokoro voices"""
        # Base IDs for different voice types
        us_female_base = "af_"  # American Female
        us_male_base = "am_"    # American Male
        uk_female_base = "bf_"  # British Female
        uk_male_base = "bm_"    # British Male
        
        # Define all possible voices
        self.VOICES = {}
        
        # US Female Voices
        us_female_names = [
            "heart", "bella", "nicole", "aoede", "kore", "sarah", 
            "nova", "sky", "alloy", "jessica", "river"
        ]
        
        # US Male Voices
        us_male_names = [
            "michael", "fenrir", "puck", "echo", "eric", 
            "liam", "onyx", "adam"
        ]
        
        # UK Female Voices
        uk_female_names = [
            "emma", "isabella", "alice", "lily"
        ]
        
        # UK Male Voices
        uk_male_names = [
            "george", "fable", "lewis", "daniel"
        ]
        
        # Add US Female voices
        for name in us_female_names:
            self.VOICES[name.capitalize()] = KokoroVoiceConfig(
                name=name.capitalize(),
                id=f"{us_female_base}{name}",
                gender="female",
                language="us",
                description=f"US English Female Voice ({name.capitalize()})"
            )
            
        # Add US Male voices
        for name in us_male_names:
            self.VOICES[name.capitalize()] = KokoroVoiceConfig(
                name=name.capitalize(),
                id=f"{us_male_base}{name}",
                gender="male",
                language="us",
                description=f"US English Male Voice ({name.capitalize()})"
            )
            
        # Add UK Female voices
        for name in uk_female_names:
            self.VOICES[name.capitalize()] = KokoroVoiceConfig(
                name=name.capitalize(),
                id=f"{uk_female_base}{name}",
                gender="female",
                language="uk",
                description=f"UK English Female Voice ({name.capitalize()})"
            )
            
        # Add UK Male voices
        for name in uk_male_names:
            self.VOICES[name.capitalize()] = KokoroVoiceConfig(
                name=name.capitalize(),
                id=f"{uk_male_base}{name}",
                gender="male",
                language="uk",
                description=f"UK English Male Voice ({name.capitalize()})"
            )
    
    def get_available_voices(self):
        """Return list of available voices"""
        if not KOKORO_AVAILABLE or not self.VOICES:
            return []
        return [v.name for v in self.VOICES.values()]
    
    def text_to_speech(self, text: str, voice_name: str, speed: float = 1.0, use_gpu: bool = False) -> str:
        """
        Convert text to speech with optimized phonemization
        
        Parameters:
        -----------
        text : str
            Input text
        voice_name : str
            Voice name
        speed : float
            Playback speed
        use_gpu : bool
            Whether to use GPU
            
        Returns:
        --------
        str
            Path to the generated audio file
        """
        if not KOKORO_AVAILABLE or not self.VOICES:
            logger.warning("Kokoro not available. Cannot generate speech.")
            return None
            
        # Extract base voice name if needed
        original_voice_name = voice_name
        if "Kokoro" in voice_name and ":" in voice_name:
            voice_name = voice_name.split(": ")[1]
        
        # Voice selection logic
        if voice_name not in self.VOICES:
            logger.warning(f"Voice '{voice_name}' not found. Using default.")
            voice_name = next(iter(self.VOICES.keys()))
            
        voice_config = self.VOICES[voice_name]
        voice_code = voice_config.id
        lang_code = 'a' if voice_config.language == 'us' else 'b'
        pipeline = self.PIPELINES[lang_code]
        
        # Clean text (remove SSML tags)
        clean_text = re.sub(r'<break\s+time="[^"]+"\s*/>', ' ', text)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)  
        clean_text = re.sub(r'\s+', ' ', clean_text.strip())
        
        logger.info(f"Generating speech with Kokoro voice: {voice_name} ({voice_code})")
        start_time = time.time()
        
        try:
            # Device selection
            device_key = False  # Default to CPU
            if use_gpu and torch.cuda.is_available():
                try:
                    with self.MODEL_LOCK:
                        if True not in self.MODEL:
                            model = KModel().to('cuda').eval()
                            # Use FP16 for improved GPU performance
                            try:
                                model = model.half()  # Use half precision for GPU
                                logger.info("Using FP16 for GPU acceleration")
                            except Exception as e:
                                logger.warning(f"FP16 conversion failed: {e}")
                            self.MODEL[True] = model
                    device_key = True
                    logger.info("Using GPU for speech synthesis")
                except Exception as e:
                    logger.error(f"GPU model load failed: {e}. Using CPU.")
            
            # Output file path
            output_file = os.path.join(tempfile.gettempdir(), f"little_geeky_speech_{int(time.time())}.wav")
            
            # Check if this voice is in cache
            if voice_name in self.VOICE_DATA_CACHE:
                voice_data = self.VOICE_DATA_CACHE[voice_name]
                logger.info(f"Using cached voice data for {voice_name}")
            else:
                # Load and cache the voice
                voice_data = pipeline.load_voice(voice_code)
                self.VOICE_DATA_CACHE[voice_name] = voice_data
                logger.info(f"Loaded and cached voice data for {voice_name}")
            
            # For short text, try to process as a single chunk first with optimized phonemizer
            if len(clean_text) < 500:
                try:
                    # Use optimized phonemizer instead of the default espeak-ng
                    phonemes = FastPhonemeProcessor.fast_phonemize(clean_text, lang_code, pipeline)
                    
                    if phonemes and len(phonemes) <= self.MAX_PHONEME_LENGTH:
                        logger.info("Using optimized phonemization for single chunk")
                        
                        # Generate audio
                        with torch.no_grad(), self.MODEL_LOCK:
                            ref_s = voice_data[len(phonemes)-1].to(self.MODEL[device_key].device)
                            audio = self.MODEL[device_key](phonemes, ref_s, speed)
                        
                        # Convert to numpy and save
                        audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                        sf.write(output_file, audio_np, 24000)
                        
                        logger.info(f"Speech generation completed in {time.time() - start_time:.2f} seconds")
                        return output_file
                except Exception as e:
                    logger.warning(f"Optimized single-chunk processing failed: {e}")
            
            # For longer text, split into chunks and process in parallel when possible
            chunks = self._split_text_into_chunks(clean_text)
            logger.info(f"Processing text in {len(chunks)} chunks with optimized phonemization")
            
            all_audio = []
            for i, chunk in enumerate(chunks):
                try:
                    # Use our optimized phonemizer
                    phonemes = FastPhonemeProcessor.fast_phonemize(chunk, lang_code, pipeline)
                    
                    if not phonemes:
                        continue
                    
                    if len(phonemes) > self.MAX_PHONEME_LENGTH:
                        phonemes = phonemes[:self.MAX_PHONEME_LENGTH]
                    
                    # Process audio with optimized settings
                    with torch.no_grad(), self.MODEL_LOCK:
                        ref_s = voice_data[len(phonemes)-1].to(self.MODEL[device_key].device)
                        audio = self.MODEL[device_key](phonemes, ref_s, speed)
                        
                    # Convert to numpy
                    audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                    all_audio.append(audio_np)
                    
                    # Free memory after processing each chunk
                    if device_key and i % 3 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
            
            # Combine audio chunks
            if not all_audio:
                logger.error("No audio generated")
                return None
                
            if len(all_audio) == 1:
                combined_audio = all_audio[0]
            else:
                combined_audio = self._combine_audio_chunks(all_audio)
            
            # Save audio
            sf.write(output_file, combined_audio, 24000)
            
            duration = time.time() - start_time
            logger.info(f"Speech generation completed in {duration:.2f} seconds")
            
            # Save phoneme cache to disk for future use
            try:
                PhonemeCache.save_cache()
            except:
                pass
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            return None
    
    def _split_text_into_chunks(self, text):
        """Split text into manageable chunks for processing with improved efficiency"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
        # Use increased target length
        target_length = self.TARGET_LENGTH
    
        chunks = []
        current_chunk = []
        current_length = 0
    
        for sentence in sentences:
            sentence_length = len(sentence)
        
            # If single sentence is too long, try to break it up
            if sentence_length > target_length:
                # Process any accumulated chunk first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
                # Try to break sentence at commas
                if ',' in sentence:
                    parts = sentence.split(',')
                    for i, part in enumerate(parts):
                        part_with_comma = part.strip() + (',' if i < len(parts) - 1 else '')
                        chunks.append(part_with_comma)
                else:
                    # If no good break point, add as is
                    chunks.append(sentence)
            
                continue
        
            # If adding this sentence would exceed target length, start new chunk
            if current_length + sentence_length > target_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length = current_length + sentence_length  # Fixed line that had syntax error
    
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
        return chunks
    
    def _combine_audio_chunks(self, audio_chunks, crossfade_ms=10):
        """Simple concatenation with minimal crossfade and optimized memory usage"""
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # Try simple concatenation first (fastest)
        try:
            return np.concatenate(audio_chunks)
        except Exception:
            pass
            
        # Fall back to crossfade if needed
        sample_rate = 24000
        fade_samples = int(crossfade_ms / 1000 * sample_rate)
        
        # Pre-calculate total length for better memory allocation
        total_length = sum(len(chunk) for chunk in audio_chunks) - fade_samples * (len(audio_chunks) - 1)
        result = np.zeros(total_length, dtype=np.float32)
        
        position = 0
        for i, chunk in enumerate(audio_chunks):
            if i == 0:
                # First chunk - no crossfade needed
                chunk_len = len(chunk)
                if chunk_len <= fade_samples:
                    result[:chunk_len] = chunk
                else:
                    result[:chunk_len - fade_samples] = chunk[:-fade_samples]
                position = chunk_len - fade_samples
            else:
                # Add crossfade
                if fade_samples > 0:
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    
                    # Make sure we're not going out of bounds
                    end_fade = min(position + fade_samples, len(result))
                    samples_to_use = end_fade - position
                    
                    if samples_to_use > 0 and samples_to_use <= len(fade_in):
                        result[position:end_fade] = (
                            result[position:end_fade] * fade_out[:samples_to_use] +
                            chunk[:samples_to_use] * fade_in[:samples_to_use]
                        )
                
                # Add rest of chunk
                chunk_remaining = chunk[fade_samples:]
                chunk_end = position + fade_samples + len(chunk_remaining)
                
                # Make sure we don't go beyond the result array
                if position + fade_samples < len(result):
                    end_pos = min(chunk_end, len(result))
                    result[position + fade_samples:end_pos] = chunk_remaining[:end_pos - (position + fade_samples)]
                
                position = position + len(chunk)
        
        return result
    
    def unload_models(self):
        """Explicitly unload models to free memory"""
        with self.MODEL_LOCK:
            self.MODEL = None
        
        # Clear voice data cache for complete cleanup
        self.VOICE_DATA_CACHE = {}
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("All models and caches unloaded from memory")
    
    def optimize_for_device(self, use_gpu=False):
        """
        Optimize the TTS engine for the specific device
        
        Args:
            use_gpu: Whether to use GPU for synthesis
            
        Returns:
            bool: Success status
        """
        try:
            # Unload any existing models
            self.unload_models()
            
            # Initialize with the appropriate device
            with self.MODEL_LOCK:
                if self.MODEL is None:
                    self.MODEL = {}
                    
                    if use_gpu and torch.cuda.is_available():
                        # GPU optimization
                        model = KModel().to('cuda').eval()
                        
                        # Try half precision for better performance
                        try:
                            model = model.half()
                            logger.info("Using FP16 precision for GPU acceleration")
                        except Exception as e:
                            logger.warning(f"FP16 conversion failed: {e}")
                            
                        self.MODEL[True] = model
                        
                        # Set CUDA optimization flags
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        
                        logger.info("Optimized for GPU operation")
                    else:
                        # CPU optimization
                        model = KModel().to('cpu').eval()
                        
                        # Try to use TorchScript optimization
                        try:
                            self.MODEL[False] = torch.jit.script(model)
                            logger.info("Using TorchScript optimization for CPU")
                        except Exception as e:
                            logger.warning(f"TorchScript optimization failed: {e}")
                            self.MODEL[False] = model
                            
                        logger.info("Optimized for CPU operation")
                
                return True
        except Exception as e:
            logger.error(f"Error optimizing for device: {e}")
            return False