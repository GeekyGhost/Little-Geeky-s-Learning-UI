# utils/text_utils.py
import re
from typing import List, Tuple, Optional

# Pre-compile regex patterns for better performance
WHITESPACE_PATTERN = re.compile(r'\s+')
NUMBER_PATTERN = re.compile(r'-?\d*\.?\d+')
HEADER_PATTERN = re.compile(r'#+(\s*)([^\n]+)')
BOLD_PATTERN = re.compile(r'\*\*([^*]+)\*\*')
ITALIC_PATTERN = re.compile(r'\*([^*]+)\*')
UNDERLINE_BOLD_PATTERN = re.compile(r'__([^_]+)__')
UNDERLINE_ITALIC_PATTERN = re.compile(r'_([^_]+)_')
LINK_PATTERN = re.compile(r'\[([^\]]+)\]\([^)]+\)')
CODE_BLOCK_PATTERN = re.compile(r'```[^`]*```')
INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
STRIKETHROUGH_PATTERN = re.compile(r'~~([^~]+)~~')
HORIZONTAL_RULE_PATTERN = re.compile(r'^\s*[\*\-_]{3,}\s*$', flags=re.MULTILINE)
LIST_MARKER_PATTERN = re.compile(r'^\s*[\*\-\+]\s+', flags=re.MULTILINE)
NUMBERED_LIST_PATTERN = re.compile(r'^\s*\d+\.\s+', flags=re.MULTILINE)

# SSML patterns
SSML_BREAK_LONG = re.compile(r'<break\s+time="[5-9][0-9][0-9]ms"\s*/>')
SSML_BREAK_MEDIUM = re.compile(r'<break\s+time="[3-4][0-9][0-9]ms"\s*/>')
SSML_BREAK_SHORT = re.compile(r'<break\s+time="[1-2][0-9][0-9]ms"\s*/>')
SSML_ANY_TAG = re.compile(r'<[^>]+>')

# Common emoji descriptions for TTS
EMOJI_DESCRIPTIONS = {
    "😊": "smile",
    "😂": "laughing",
    "❤️": "heart",
    "👍": "thumbs up",
    "🙏": "prayer hands",
    "😃": "grinning",
    "😉": "winking",
    "😢": "crying",
    "😎": "cool",
    "🤔": "thinking"
}

# Common contractions for natural speech
CONTRACTIONS = {
    "let's": "lets",
    "can't": "cant",
    "won't": "wont",
    "don't": "dont",
    "isn't": "isnt",
    "didn't": "didnt",
    "hasn't": "hasnt",
    "haven't": "havent",
    "shouldn't": "shouldnt",
    "wouldn't": "wouldnt",
    "couldn't": "couldnt",
    "it's": "its",
    "that's": "thats",
    "there's": "theres",
    "he's": "hes",
    "she's": "shes",
    "what's": "whats",
    "where's": "wheres",
    "who's": "whos",
    "how's": "hows",
    "when's": "whens",
    "why's": "whys",
    "i'm": "im",
    "you're": "youre",
    "we're": "were",
    "they're": "theyre"
}

def extract_number(text: str) -> Optional[float]:
    """Extract numerical value from text, handling various formats."""
    if text is None:
        return None
        
    text = text.strip().replace('$', '').replace('£', '').replace('€', '')
    matches = NUMBER_PATTERN.findall(text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None

def clean_markdown(text: str) -> str:
    """Clean markdown formatting from text."""
    # Handle headers
    text = HEADER_PATTERN.sub(r'\2. ', text)
    
    # Handle bold/italic formatting
    text = BOLD_PATTERN.sub(r'\1', text)
    text = ITALIC_PATTERN.sub(r'\1', text) 
    text = UNDERLINE_BOLD_PATTERN.sub(r'\1', text)
    text = UNDERLINE_ITALIC_PATTERN.sub(r'\1', text)
    
    # Handle links
    text = LINK_PATTERN.sub(r'\1', text)
    
    # Handle code blocks and inline code
    text = CODE_BLOCK_PATTERN.sub(' ', text)
    text = INLINE_CODE_PATTERN.sub(r'\1', text)
    
    # Handle strikethrough
    text = STRIKETHROUGH_PATTERN.sub(r'\1', text)
    
    # Handle horizontal rules
    text = HORIZONTAL_RULE_PATTERN.sub(' ', text)
    
    # Handle lists
    text = LIST_MARKER_PATTERN.sub('Item: ', text)
    text = NUMBERED_LIST_PATTERN.sub('Item: ', text)
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences based on punctuation and new lines."""
    # First, handle new lines as potential sentence boundaries
    lines = text.split('\n')
    raw_sentences = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Split on sentence-ending punctuation
        line_sentences = re.split(r'(?<=[.!?])\s+', line)
        raw_sentences.extend([s.strip() for s in line_sentences if s.strip()])
    
    # Post-process sentences to ensure they're meaningful
    sentences = []
    current = ""
    
    for sentence in raw_sentences:
        # If very short, might be a fragment; combine with next
        if len(sentence) < 5 and current and not any(p in current[-1] for p in '.!?'):
            current += " " + sentence
        else:
            if current:
                sentences.append(current)
            current = sentence
    
    if current:
        sentences.append(current)
        
    return sentences

def format_text_for_speech(text: str, for_speech: bool = True, add_ssml: bool = True) -> str:
    """
    Comprehensive text processing for TTS preparation.
    
    Args:
        text: Input text to process
        for_speech: Whether to optimize for speech (vs display)
        add_ssml: Whether to add SSML tags for pauses and emphasis
        
    Returns:
        Processed text ready for TTS
    """
    if not text:
        return ""

    # Step 1: Basic cleanup
    text = WHITESPACE_PATTERN.sub(' ', text.strip())
    text = text.replace('…', '...').replace('–', '-').replace('—', '-')
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # Step 2: Remove markdown formatting
    text = clean_markdown(text)
    
    # Step 3: Process sentences and ensure coherent paragraphs
    sentences = split_into_sentences(text)
    processed_text = []
    
    for sentence in sentences:
        # Ensure sentence ends with punctuation
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        processed_text.append(sentence)
    
    # Step 4: Add SSML if requested
    if for_speech and add_ssml:
        result = []
        for sentence in processed_text:
            # Add appropriate pauses
            sentence = re.sub(r'(\.\s)', '. <break time="500ms"/> ', sentence)
            sentence = re.sub(r'(!\s)', '! <break time="500ms"/> ', sentence)
            sentence = re.sub(r'(\?\s)', '? <break time="500ms"/> ', sentence)
            sentence = re.sub(r'(,\s)', ', <break time="250ms"/> ', sentence)
            sentence = re.sub(r'(;\s)', '; <break time="350ms"/> ', sentence)
            sentence = re.sub(r'Item:\s', 'Item: <break time="350ms"/> ', sentence)
            result.append(sentence)
        
        return ' <break time="750ms"/> '.join(result)
    else:
        return '\n\n'.join(processed_text)

def prepare_for_tts(text: str, engine: str = "general") -> str:
    """
    Prepare text for TTS with engine-specific optimizations.
    
    Args:
        text: Input text to process
        engine: TTS engine type ('general', 'kokoro', 'pyttsx3')
        
    Returns:
        Optimized text for the specified TTS engine
    """
    # Basic formatting for all engines
    text = format_text_for_speech(text, for_speech=True)
    
    # Apply engine-specific processing
    if engine == "pyttsx3":
        # pyttsx3 doesn't support SSML, so remove all tags
        text = SSML_ANY_TAG.sub('', text)
    elif engine == "kokoro":
        # Kokoro has different pause handling
        text = SSML_BREAK_LONG.sub('.\n\n', text)
        text = SSML_BREAK_MEDIUM.sub('; ', text)
        text = SSML_BREAK_SHORT.sub(', ', text)
        
        # Apply contractions for better pronunciation in Kokoro
        for contraction, replacement in CONTRACTIONS.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def split_text_into_chunks(text: str, max_chars: int = 500) -> List[str]:
    """
    Split text into appropriately sized chunks for processing.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Get sentences first
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If sentence alone exceeds max length, split it further
        if sentence_length > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_length = 0
                
            # Split long sentence on logical breaks
            for delimiter in ['. ', '? ', '! ', '; ', ': ', ', ']:
                if delimiter in sentence:
                    parts = sentence.split(delimiter)
                    for i, part in enumerate(parts):
                        part_with_delimiter = part + delimiter if i < len(parts) - 1 else part
                        part_length = len(part_with_delimiter)
                        
                        if current_length + part_length <= max_chars:
                            current_chunk += part_with_delimiter
                            current_length += part_length
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = part_with_delimiter
                            current_length = part_length
                    
                    if current_chunk:
                        break
            else:
                # If no delimiters found, split by character count
                for i in range(0, sentence_length, max_chars):
                    chunks.append(sentence[i:i+max_chars])
        
        # Normal case - add sentence if it fits
        elif current_length + sentence_length <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
            current_length += sentence_length
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_length = sentence_length
    
    # Add the last chunk if there is one
    if current_chunk:
        chunks.append(current_chunk)
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def replace_emojis(text: str) -> str:
    """Replace common emojis with their text descriptions."""
    for emoji, description in EMOJI_DESCRIPTIONS.items():
        text = text.replace(emoji, f" {description} ")
    return text