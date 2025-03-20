# services/ollama_service.py
import aiohttp
from typing import List
from utils.logging_utils import logger
from config.settings import OLLAMA_API_URL

class OllamaClient:
    def __init__(self, api_url: str = OLLAMA_API_URL):
        self.api_url = api_url

    async def get_models_async(self) -> List[str]:
        """Get list of available models from Ollama API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'models' in data and data['models']:
                            models = [model['name'] for model in data['models']]
                            logger.info(f"Found {len(models)} models: {', '.join(models)}")
                            return models
                        else:
                            logger.warning("No models found in Ollama API response")
                    else:
                        logger.error(f"Ollama API returned status code: {response.status}")
                    return ["No models found 😕"]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return ["Please check if Ollama is running 🤔"]

    async def generate_response_async(self, model: str, prompt: str, context: str = "") -> str:
        """Generate a response from the Ollama API"""
        try:
            formatted_prompt = f"{context}\n{prompt}" if context else prompt
            data = {
                "model": model,
                "prompt": formatted_prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/generate", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("response", "I couldn't think of anything 😕").strip()
                        return text
                    else:
                        logger.error(f"Error from Ollama API: {response.status}")
                        return f"Sorry, I had trouble thinking! Status code: {response.status}"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I'm having trouble thinking right now! 🤔"
            
    async def chat_with_images(self, model: str, prompt: str, image_data: str) -> str:
        """Send chat request with image data to the Ollama API"""
        try:
            if not model:
                logger.error("No model specified for chat_with_images")
                return "Error: No model specified for image analysis"
                
            logger.info(f"Processing image with model: {model}")
            
            request_data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "message" in result:
                            return result["message"]["content"]
                    
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    return f"Error: Failed to analyze image. API returned: {error_text}"
        except Exception as e:
            logger.error(f"Error in chat with images: {e}")
            return f"Error in image analysis: {str(e)}"