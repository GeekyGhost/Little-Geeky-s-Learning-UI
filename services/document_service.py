# services/document_service.py
import os
import tempfile
import base64
import io
from PIL import Image
import fitz  # PyMuPDF
from utils.logging_utils import logger
from services.ollama_service import OllamaClient
from config.settings import MAX_IMAGE_SIZE_MB
from utils.settings_utils import SettingsManager

class DocumentProcessor:
    """Handles document processing for PDFs and images"""
    def __init__(self):
        try:
            os.makedirs('temp', exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating temp directory: {e}")
    
    @staticmethod
    def process_document(file_path):
        """Process document file (PDF or image)"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                pages = []
                with fitz.open(file_path) as pdf:
                    for page_num in range(len(pdf)):
                        page = pdf[page_num]
                        # Create a temporary file for the page image
                        temp_img_path = os.path.join(tempfile.gettempdir(), f"page_{page_num}.png")
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        pix.save(temp_img_path)
                        
                        pages.append({
                            'type': 'pdf_page',
                            'content': page.get_text(),
                            'path': temp_img_path,
                            'original_path': file_path,
                            'page_number': page_num + 1,
                            'total_pages': len(pdf)
                        })
                return pages
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                # Process single image
                return [{
                    'type': 'image',
                    'content': '',  # Will be set after image description
                    'path': file_path,
                    'page_number': 1,
                    'total_pages': 1
                }]
            
            return []
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []

    @staticmethod
    def cleanup_temp_files(documents):
        """Clean up temporary files created during document processing."""
        for doc in documents:
            if doc['type'] == 'pdf_page' and os.path.exists(doc['path']):
                try:
                    os.remove(doc['path'])
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {doc['path']}: {e}")

    @staticmethod
    async def describe_image(image_path, ollama_client, model=None):
        """Generate a description of an image using the Ollama API"""
        try:
            if not os.path.exists(image_path):
                return "Error: Image file not found"

            file_size = os.path.getsize(image_path) / (1024 * 1024)
            if file_size > MAX_IMAGE_SIZE_MB:
                return f"Error: Image file too large (max {MAX_IMAGE_SIZE_MB}MB)"

            # Use settings manager to get the appropriate vision model if not specified
            if not model:
                model = SettingsManager.get_model_for_task("vision")
                logger.info(f"Using vision model from settings: {model}")
            
            # Encode image to base64
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                image_data = base64.b64encode(image_bytes).decode()

            prompt = """Please provide a detailed analysis of this image with the following structure:
            1. Main Subject: What is the primary focus or subject of the image?
            2. Visual Elements: What notable features, objects, or details are present?
            3. Colors and Style: Describe the color scheme, lighting, and artistic style
            4. Text/Symbols: Note any visible text, logos, or symbolic elements
            5. Context: What is the apparent purpose or context of this image?

            Please make sure to be accurate and specific in your description."""
            
            # Call the chat_with_images method
            response = await ollama_client.chat_with_images(model, prompt, image_data)
            return response

        except Exception as e:
            logger.error(f"Error describing image: {e}")
            return f"Error describing image: {str(e)}"