# services/audio_service.py
import os
import wave
import time
import queue
import pyaudio
import tempfile
import threading
import pyttsx3
import speech_recognition as sr
from dataclasses import dataclass
from typing import List, Tuple, Optional
import concurrent.futures
from utils.logging_utils import logger
from utils.text_utils import format_text_for_speech
import re

# Try to import torch
try:
    import torch
except ImportError:
    pass

# Import Kokoro service
try:
    from services.kokoro_service import KokoroTTS, KOKORO_AVAILABLE
except ImportError as e:
    KOKORO_AVAILABLE = False
    logger.error(f"Kokoro service import error: {e}")

@dataclass
class VoiceConfig:
    """Voice configuration for text-to-speech"""
    name: str
    id: str
    gender: str
    age: Optional[str] = None
    description: Optional[str] = None
    engine: str = "pyttsx3"  # "kokoro" or "pyttsx3"

class AudioProcessor:
    """Handles text-to-speech processing"""
    def __init__(self):
        # Initialize pyttsx3 as fallback
        self.engine = pyttsx3.init(driverName='sapi5' if os.name == 'nt' else None)
        self.voice_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Try to initialize Kokoro TTS
        self.kokoro_tts = None
        self.uses_kokoro = False
        if KOKORO_AVAILABLE:
            try:
                self.kokoro_tts = KokoroTTS()
                self.uses_kokoro = bool(self.kokoro_tts.VOICES)
                logger.info(f"Successfully initialized Kokoro TTS with {len(self.kokoro_tts.VOICES)} voices")
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro TTS: {e}")
                self.uses_kokoro = False
        
        # Load voice configurations
        self.voices = self._initialize_voices()
        
    def _initialize_voices(self) -> List[VoiceConfig]:
        """Initialize available voices"""
        voice_configs = []
        
        # First try to get Kokoro voices
        if self.uses_kokoro and self.kokoro_tts and hasattr(self.kokoro_tts, 'VOICES'):
            try:
                logger.info(f"Loading Kokoro voices. Available: {list(self.kokoro_tts.VOICES.keys())}")
                
                # Add Kokoro voices
                for voice_name, voice_config in self.kokoro_tts.VOICES.items():
                    gender_symbol = '(F)' if voice_config.gender == 'female' else '(M)'
                    lang_symbol = 'US' if voice_config.language == 'us' else 'UK'
                    
                    display_name = f"Kokoro {lang_symbol} {gender_symbol}: {voice_name}"
                    
                    config = VoiceConfig(
                        name=display_name,
                        id=voice_config.id,
                        gender=voice_config.gender,
                        description=voice_config.description,
                        engine="kokoro"
                    )
                    voice_configs.append(config)
                    logger.info(f"Added Kokoro voice: {display_name} with ID {voice_config.id}")
                
                if voice_configs:
                    logger.info(f"Loaded {len(voice_configs)} Kokoro voices")
            except Exception as e:
                logger.error(f"Error loading Kokoro voices: {e}")
                
        # Always add pyttsx3 voices as fallback
        try:
            for voice in self.engine.getProperty('voices'):
                desc = voice.name.lower()
                gender = "female" if "female" in desc else "male"
                age = "child" if "child" in desc else "adult"
                gender_symbol = '(F)' if gender == 'female' else '(M)'
                
                config = VoiceConfig(
                    name=f"System {gender_symbol}: {voice.name}",
                    id=voice.id,
                    gender=gender,
                    age=age,
                    description=voice.name,
                    engine="pyttsx3"
                )
                voice_configs.append(config)
            
            logger.info(f"Loaded {len(voice_configs) - (len(self.kokoro_tts.VOICES) if self.uses_kokoro and self.kokoro_tts and hasattr(self.kokoro_tts, 'VOICES') else 0)} system voices")
        except Exception as e:
            logger.error(f"Error initializing system voices: {e}")
            # Fallback to basic voice
            if not voice_configs:
                try:
                    default_voice = self.engine.getProperty('voices')[0]
                    voice_configs.append(VoiceConfig(
                        name=f"System Default: {default_voice.name}",
                        id=default_voice.id,
                        gender="unknown",
                        description="Default System Voice",
                        engine="pyttsx3"
                    ))
                except Exception as e2:
                    logger.error(f"Error getting default voice: {e2}")
                    voice_configs.append(VoiceConfig(
                        name="System Default",
                        id="default",
                        gender="unknown",
                        description="Default System Voice",
                        engine="pyttsx3"
                    ))
            
        return voice_configs

    def text_to_speech_async(self, text: str, voice_id: str, speed: float) -> str:
        """Convert text to speech asynchronously"""
        future = self.executor.submit(self._text_to_speech, text, voice_id, speed)
        return future.result()

    def _text_to_speech(self, text: str, voice_id: str, speed: float) -> str:
        """Internal method to convert text to speech"""
        try:
            logger.info(f"Converting text to speech with voice ID: {voice_id}")
            
            # Clean potential SSML tags for pyttsx3 (which doesn't support them)
            clean_text = self._clean_ssml_tags(text)
            
            # Format text with proper pauses - don't use with pyttsx3 since it doesn't support SSML
            formatted_text_for_kokoro = format_text_for_speech(text)
            
            # Find the selected voice's configuration
            selected_voice = None
            for voice in self.voices:
                if voice.id == voice_id:
                    selected_voice = voice
                    logger.info(f"Selected voice: {voice.name}, Engine: {voice.engine}")
                    break
            
            if not selected_voice:
                logger.warning(f"Voice ID {voice_id} not found, using default")
                return self._use_pyttsx3(clean_text, voice_id, speed)
            
            # Check which engine to use
            if selected_voice.engine == "kokoro" and self.uses_kokoro and self.kokoro_tts:
                # Extract the Kokoro voice name from the display name
                if "Kokoro" in selected_voice.name and ":" in selected_voice.name:
                    kokoro_voice_name = selected_voice.name.split(": ")[1]
                    logger.info(f"Using Kokoro voice: {kokoro_voice_name}")
                    
                    # Now check if this voice exists in the Kokoro voices dictionary
                    if kokoro_voice_name in self.kokoro_tts.VOICES:
                        output_file = self.kokoro_tts.text_to_speech(
                            formatted_text_for_kokoro,  # Use SSML-formatted text for Kokoro
                            kokoro_voice_name,
                            speed,
                            use_gpu=False  # Default to CPU for compatibility
                        )
                        if output_file:
                            logger.info(f"Successfully generated speech with Kokoro, file: {output_file}")
                            return output_file
                    else:
                        logger.warning(f"Kokoro voice {kokoro_voice_name} not found in available voices")
                else:
                    logger.warning(f"Could not extract Kokoro voice name from {selected_voice.name}")
            
            # Fall back to pyttsx3
            logger.info("Falling back to pyttsx3")
            return self._use_pyttsx3(clean_text, voice_id, speed)
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            try:
                # Last resort fallback
                return self._use_pyttsx3(clean_text, None, speed)
            except:
                return None
    
    def _clean_ssml_tags(self, text):
        """Remove SSML tags for engines that don't support them"""
        # Remove break tags
        cleaned = re.sub(r'<break\s+time="[^"]+"\s*/>', '', text)
        # Remove any other XML-like tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned
    
    def _use_pyttsx3(self, text, voice_id, speed):
        """Use pyttsx3 for TTS"""
        try:
            original_rate = self.engine.getProperty('rate')
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            self.engine.setProperty('rate', int(original_rate * speed))
            
            output_file = os.path.join(tempfile.gettempdir(), f"little_geeky_speech_{int(time.time())}.mp3")
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            # Reset rate after speaking
            self.engine.setProperty('rate', original_rate)
            return output_file
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return None

class VoiceRecorder:
    """Handles voice recording and recognition"""
    def __init__(self):
        self.recording_state = {
            'is_recording': False,
            'frames': [],
            'stream': None
        }

    def toggle_recording(self) -> Tuple[bool, List[bytes]]:
        """Toggle recording state"""
        if not self.recording_state['is_recording']:
            return self._start_recording()
        else:
            return self._stop_recording()

    def _start_recording(self) -> Tuple[bool, None]:
        """Start recording audio"""
        try:
            self.recording_state['is_recording'] = True
            self.recording_state['frames'] = []
            p = pyaudio.PyAudio()
            self.recording_state['stream'] = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )

            def record():
                while self.recording_state['is_recording']:
                    try:
                        data = self.recording_state['stream'].read(1024, exception_on_overflow=False)
                        self.recording_state['frames'].append(data)
                    except Exception as e:
                        logger.error(f"Recording error: {e}")
                        break

            threading.Thread(target=record, daemon=True).start()
            return True, None
        except Exception as e:
            logger.error(f"Couldn't start recording: {e}")
            self.recording_state['is_recording'] = False
            return False, None

    def _stop_recording(self) -> Tuple[bool, List[bytes]]:
        """Stop recording audio"""
        if not self.recording_state['is_recording']:
            return False, []

        try:
            self.recording_state['is_recording'] = False
            if self.recording_state['stream']:
                self.recording_state['stream'].stop_stream()
                self.recording_state['stream'].close()
            return False, self.recording_state['frames']
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False, []
            
    def process_recording(self, frames) -> str:
        """Process recorded audio to text"""
        if not frames:
            return ""
            
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b"".join(frames))

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file.name) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass