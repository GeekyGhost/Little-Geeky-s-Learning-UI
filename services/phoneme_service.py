# services/phoneme_service.py
import os
import re
import json
import hashlib
import threading
from typing import Dict, Tuple, List, Optional
from utils.logging_utils import logger
from config.settings import DATA_DIR

class PhonemeCache:
    """
    A high-performance phoneme caching system to bypass the slow espeak-ng phonemizer
    when possible for common text patterns.
    """
    
    # Class-level cache with thread safety
    _cache = {}
    _cache_lock = threading.Lock()
    _cache_file = os.path.join(DATA_DIR, "phoneme_cache.json")
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Load the phoneme cache from disk if it exists"""
        if cls._initialized:
            return
            
        try:
            if os.path.exists(cls._cache_file):
                with open(cls._cache_file, 'r', encoding='utf-8') as f:
                    cls._cache = json.load(f)
                logger.info(f"Loaded {len(cls._cache)} cached phoneme entries")
            cls._initialized = True
        except Exception as e:
            logger.error(f"Error loading phoneme cache: {e}")
            cls._cache = {}
            cls._initialized = True
    
    @classmethod
    def save_cache(cls):
        """Save the current cache to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cls._cache_file), exist_ok=True)
            
            with open(cls._cache_file, 'w', encoding='utf-8') as f:
                json.dump(cls._cache, f)
            logger.info(f"Saved {len(cls._cache)} entries to phoneme cache")
        except Exception as e:
            logger.error(f"Error saving phoneme cache: {e}")
    
    @classmethod
    def get_cached_phonemes(cls, text: str, lang_code: str) -> Optional[List[int]]:
        """
        Get cached phonemes for a given text and language code
        
        Args:
            text: The text to get phonemes for
            lang_code: The language code ('a' for US English, 'b' for UK English)
            
        Returns:
            List of phoneme IDs if cached, None otherwise
        """
        cls.initialize()
        
        # Create a cache key using a hash of the text and language
        cache_key = f"{hashlib.md5((text + lang_code).encode()).hexdigest()}"
        
        with cls._cache_lock:
            return cls._cache.get(cache_key)
    
    @classmethod
    def add_to_cache(cls, text: str, lang_code: str, phonemes: List[int]):
        """
        Add phonemes to the cache
        
        Args:
            text: The text to cache phonemes for
            lang_code: The language code
            phonemes: The phoneme IDs to cache
        """
        cls.initialize()
        
        # Create a cache key using a hash of the text and language
        cache_key = f"{hashlib.md5((text + lang_code).encode()).hexdigest()}"
        
        with cls._cache_lock:
            cls._cache[cache_key] = phonemes
            
            # Save the cache if it grows significantly
            if len(cls._cache) % 100 == 0:
                cls.save_cache()

class FastPhonemeProcessor:
    """
    A faster alternative to the espeak-ng phonemizer when possible,
    with fallback to the standard pipeline for unfamiliar text.
    """
    
    # Common English phoneme mappings for frequent words and patterns
    # These are approximate but work well enough for common words
    COMMON_PHONEMES = {
        'a': [43],  # /ə/ or /eɪ/ depending on context
        'the': [55, 46],  # /ðə/
        'to': [53, 56],  # /tu/
        'and': [61, 43, 48],  # /ænd/
        'of': [43, 62],  # /əv/
        'in': [47, 48],  # /ɪn/
        'is': [47, 58],  # /ɪz/
        'it': [47, 53],  # /ɪt/
        'you': [54, 56],  # /ju/
        'for': [61, 59],  # /fɔː/
        'that': [55, 61, 53],  # /ðæt/
        'with': [52, 47, 55],  # /wɪð/
        'this': [55, 47, 58],  # /ðɪs/
        'are': [46, 59],  # /ɑː/
        'on': [43, 48],  # /ɒn/
        'at': [61, 53],  # /æt/
        'by': [42, 49],  # /baɪ/
        'be': [42, 51],  # /biː/
        'or': [59, 59],  # /ɔː/
        'not': [48, 43, 53],  # /nɒt/
        'as': [61, 58],  # /æz/
        'from': [61, 50, 43, 50],  # /frʌm/
        'an': [43, 48],  # /æn/
        'was': [52, 43, 58],  # /wɒz/
        'have': [47, 61, 62],  # /hæv/
        'has': [47, 61, 58],  # /hæz/
        'can': [49, 43, 48],  # /kæn/
        'will': [52, 47, 44],  # /wɪl/
        'would': [52, 51, 44],  # /wʊd/
        'should': [55, 51, 44],  # /ʃʊd/
        'could': [49, 51, 44],  # /kʊd/
        'they': [55, 51],  # /ðeɪ/
        'them': [55, 47, 50],  # /ðəm/
        'their': [55, 51, 59],  # /ðeə/
        'there': [55, 51, 59],  # /ðeə/
        'these': [55, 51, 58],  # /ðiːz/
        'those': [55, 47, 58],  # /ðəʊz/
        'what': [52, 43, 53],  # /wɒt/
        'when': [52, 47, 48],  # /wen/
        'where': [52, 51, 50],  # /weə/
        'which': [52, 47, 45],  # /wɪtʃ/
        'who': [47, 56],  # /huː/
        'why': [52, 49],  # /waɪ/
        'how': [47, 57],  # /haʊ/
        'we': [52, 51],  # /wiː/
        'us': [43, 58],  # /ʌs/
        'our': [57, 59],  # /aʊə/
        'me': [50, 51],  # /miː/
        'my': [50, 49],  # /maɪ/
        'your': [54, 59, 59],  # /jɔː/
        'he': [47, 51],  # /hiː/
        'she': [55, 51],  # /ʃiː/
        'it': [47, 53],  # /ɪt/
        'here': [47, 46],  # /hɪə/
        'now': [48, 57],  # /naʊ/
        'then': [55, 47, 48],  # /ðen/
        'some': [58, 43, 50],  # /sʌm/
        'all': [44, 44],  # /ɔːl/
        'one': [52, 43, 48],  # /wʌn/
        'two': [53, 56],  # /tuː/
        'three': [46, 51],  # /θriː/
        'like': [44, 49, 46],  # /laɪk/
        'time': [53, 49, 50],  # /taɪm/
        'day': [44, 51],  # /deɪ/
        'out': [57, 53],  # /aʊt/
        'up': [43, 50],  # /ʌp/
        'down': [44, 57, 48],  # /daʊn/
        'over': [47, 52, 46],  # /əʊvə/
        'about': [43, 42, 57, 53],  # /əˈbaʊt/
        'through': [46, 56],  # /θruː/
        'during': [44, 56, 46, 47, 48],  # /ˈdjʊərɪŋ/
        'before': [42, 51, 59, 59],  # /bɪˈfɔː/
        'after': [46, 59, 53, 46],  # /ˈɑːftə/
        'between': [42, 51, 53, 52, 51, 48],  # /bɪˈtwiːn/
        'against': [43, 45, 47, 48, 58, 53],  # /əˈɡenst/
    }
    
    @classmethod
    def fast_phonemize(cls, text: str, lang_code: str, pipeline) -> Optional[List[int]]:
        """
        Attempt to quickly phonemize the text using cached or common patterns,
        falling back to the standard pipeline for unfamiliar text.
        
        Args:
            text: The text to phonemize
            lang_code: The language code 
            pipeline: The Kokoro pipeline to use as fallback
            
        Returns:
            List of phoneme IDs, or None if phonemization failed
        """
        if not text:
            return None
            
        # First check if we have the exact text in cache
        cached = PhonemeCache.get_cached_phonemes(text, lang_code)
        if cached is not None:
            return cached
            
        # For very short text (single words or short phrases)
        if len(text) < 30:
            # Try exact match in common phonemes dict
            text_lower = text.lower().strip()
            if text_lower in cls.COMMON_PHONEMES:
                return cls.COMMON_PHONEMES[text_lower]
                
            # For single words, still use the cache but fallback to normal phonemization
            try:
                phonemes, _ = pipeline.g2p(text)
                if phonemes:
                    # Cache the result for future use
                    PhonemeCache.add_to_cache(text, lang_code, phonemes)
                return phonemes
            except Exception as e:
                logger.warning(f"Phonemization failed for short text: {e}")
                return None
                
        # For longer text, split into words and try to phonemize each
        # This is much faster for common text patterns
        try:
            # Clean and tokenize the text
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Build phoneme sequence from known words where possible
            combined_phonemes = []
            fallback_needed = False
            
            for word in words:
                if word in cls.COMMON_PHONEMES:
                    combined_phonemes.extend(cls.COMMON_PHONEMES[word])
                else:
                    fallback_needed = True
                    break
            
            # If we have all needed phonemes, use them
            if not fallback_needed and combined_phonemes:
                # Add the result to cache
                PhonemeCache.add_to_cache(text, lang_code, combined_phonemes)
                return combined_phonemes
            
            # Otherwise, fall back to standard pipeline
            phonemes, _ = pipeline.g2p(text)
            if phonemes:
                # Cache the result for future use
                PhonemeCache.add_to_cache(text, lang_code, phonemes)
            return phonemes
            
        except Exception as e:
            logger.warning(f"Failed to phonemize text: {e}")
            
            # Last resort: try the standard pipeline directly
            try:
                phonemes, _ = pipeline.g2p(text)
                return phonemes
            except:
                return None