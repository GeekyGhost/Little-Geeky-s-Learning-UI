# Little Geeky's Learning Adventure

<img width="772" alt="reading" src="https://github.com/user-attachments/assets/d956b3f9-8a38-493d-85bb-9f667493c854" />


<img width="388" alt="math" src="https://github.com/user-attachments/assets/d70aa3e4-fc06-4671-8fa8-f8233b32ac35" />


<img width="577" alt="typing" src="https://github.com/user-attachments/assets/de10815e-1cb6-4b9f-864a-a6d0a0ff27e1" />



<img width="1146" alt="Screenshot 2025-03-29 151128" src="https://github.com/user-attachments/assets/851621d3-caae-4aff-ba1e-398fbb3095a1" />



> ⚠️ **IMPORTANT SAFETY NOTICE** ⚠️
> 
> The Image Creator feature does not currently implement NSFW filtering. This feature is planned but not yet integrated.
> Adult supervision is strongly recommended when children use the Image Creator.
> This module is still in development - consider this a stable beta version that requires further safety enhancements.

## About

Little Geeky's Learning Adventure is an interactive educational platform designed to make learning engaging and fun for elementary school children (grades 1-6). This application combines cutting-edge AI technologies with research-backed educational methodologies to create a supportive, adaptive learning environment.

Developed as a passion project by a single developer, Little Geeky represents countless hours of development, research, and testing to create a tool that can help children develop crucial skills in reading, mathematics, typing, and now creative image generation.

## Features

- **AI-Powered Learning**: Leverages local large language models (via Ollama) to provide personalized, contextually relevant educational content
- **Multi-Modal Learning**: Supports text, audio, image-based learning, and creative image generation experiences
- **Voice Interaction**: High-quality text-to-speech capabilities bring content to life
- **Achievement System**: Motivates learners with achievement tracking and progress visualization
- **Customizable Experience**: Settings for voice, speed, theme, and more
- **Educational Focus Areas**: Reading comprehension, mathematics, typing skills, and creative image creation

## New! Image Creator

The application now includes an Image Creator feature:

> ⚠️ **Safety Notice**: Adult supervision required. No content filtering system is currently implemented.

- **Create Images from Text**: Generate custom images based on text descriptions
- **Customization Options**: Adjust settings like image size, generation steps, and guidance scale
- **Style Templates**: Quick access to different artistic styles like Cartoon, Watercolor, 3D Render, and Pixel Art
- **Image Saving**: Save your generated images to reuse in educational activities
- **Transparent Background Option**: Create images with transparent backgrounds for use in other applications
- **Safety Considerations**: This feature is in beta. Always supervise children during use, as content filtering is still in development

## Installation

### Prerequisites

- **Python 3.8+**: Required for the application's core functionality
- **Ollama**: To run local LLMs (Large Language Models)
- **Windows, macOS, or Linux**: Supported operating systems
- **Stable Diffusion Models**: Required for the Image Creator feature

### Detailed Installation Steps

#### 1. Setting Up Ollama (Required for AI Features)

Little Geeky uses Ollama for AI capabilities. To install Ollama:

- **Windows/macOS/Linux**: Download and install from [Ollama's official website](https://ollama.ai/download)
- Once installed, run Ollama and keep it running in the background while using Little Geeky

#### 2. Installing Little Geeky

**Option 1: Using the provided setup script**

1. Clone the repository or download the ZIP file
   ```
   git clone https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git
   cd Little-Geeky-s-Learning-UI
   ```

2. Run the setup script:
   - **Windows**: Double-click `run.bat` or run it from the command line
   - **macOS/Linux**: Make the script executable and run it
     ```
     chmod +x run.sh
     ./run.sh
     ```

**Option 2: Manual installation**

1. Clone the repository
   ```
   git clone https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git
   cd Little-Geeky-s-Learning-UI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```
   python main.py
   ```

#### 3. First-time Setup

When launching Little Geeky for the first time:

1. The application will create necessary directories for storing user data and models
2. For the speech synthesis to work properly:
   - **Kokoro TTS**: Will be installed with the requirements, but may need additional dependencies:
     - Windows: Visual C++ Redistributable might be required
     - Linux: Additional audio libraries might be needed (`libsndfile1`)
   - **System voices**: Will use your OS's available voices

#### 4. Model Setup in Ollama

For optimal performance, download these models in Ollama:

1. Open a terminal/command prompt and run:
   ```
   # Basic model for text generation (choose one)
   ollama pull llama3
   
   # Vision model for document analysis (choose one)
   ollama pull llava
   ```

2. Within Little Geeky, go to the Settings tab → Model Management to set up these models for different tasks

#### 5. Installing Stable Diffusion Models for Image Creator

To use the Image Creator feature, you need to download Stable Diffusion models:

1. Create a `Checkpoints` folder in the application directory if it doesn't already exist
2. Download the GeekyGhost LCM model from [Civitai](https://civitai.com/models/476202/geekyghost-lcm)
3. Place the downloaded model file in the `Checkpoints` folder
4. The model will be automatically detected when you use the Image Creator tab

#### 6. Directory Structure

After installation, these directories will be created:

- `data/` - Stores user progress and settings
- `logs/` - Contains application logs for troubleshooting
- `models/` - Used for caching model information
- `temp/` - Temporary files for document processing
- `assets/images/` - Store generated images
- `Checkpoints/` - Storage for Stable Diffusion models

### System Requirements

- **Minimum**:
  - CPU: Dual-core processor
  - RAM: 4GB (8GB recommended if using large AI models)
  - Storage: 500MB for the application + space for models (models can range from 1GB to 8GB each)
  - Network: Not required after installation
  
- **Recommended** (for optimal performance):
  - CPU: Quad-core processor
  - RAM: 16GB or more
  - GPU: NVIDIA GPU with CUDA support (for faster AI processing and image generation)
  - Storage: SSD with at least 20GB free space

### Troubleshooting

- **Application doesn't start**:
  - Check Python version (`python --version`)
  - Ensure virtual environment is activated
  - Look at logs in the `logs/` directory

- **No AI responses**:
  - Verify Ollama is running in the background
  - Check models are downloaded in Ollama
  - Configure models in Settings tab

- **No audio output**:
  - Check system audio settings
  - Ensure audio libraries are installed
  - Try switching to system voices in the settings

- **Image Creator doesn't work**:
  - Verify you have models in the `Checkpoints` folder
  - Check for GPU memory issues if you have a GPU
  - Try lowering image dimensions or steps in the UI

- **For log file access**:
  - Look in `logs/little_geeky.log` for detailed error messages

## Interface & Navigation

Little Geeky features a tab-based interface with the following main areas:

- **Reading Tab**: For reading practice, document analysis, and comprehension
- **Math Tab**: For mathematics exercises tailored to different grade levels
- **Typing Tab**: For keyboard skills development and practice
- **Image Creator Tab**: For generating custom images from text descriptions
- **Achievements Tab**: For tracking learning progress and accomplishments
- **Settings Tab**: For customizing the learning experience

## Image Creator Guide

The Image Creator tab provides a creative environment where children can turn their ideas into images.

> ⚠️ **SAFETY INFORMATION**: The current implementation does not include content filtering systems. Adult supervision is required. This feature is still in development and should be considered a beta version.

**Features:**
- **Text-to-Image Generation**: Create images from text descriptions
- **Negative Prompt**: Specify elements to avoid in the generated image
- **Generation Options**: Control image size, quality, and style
- **Style Templates**: Quick selection of artistic styles
- **Image Saving**: Save creations to the assets folder
- **Safeguards**: While technical safety features are being developed, use child-friendly prompts and templates

**How to Use:**
1. Enter a description of the image you want to create in the text field
2. Optionally, add negative prompts to avoid certain elements
3. Select the generation speed (LCM for faster results, Standard for higher quality)
4. Adjust steps, guidance scale, and dimensions as needed
5. Choose a style template or customize settings in the Advanced section
6. Click "Generate Image" to create your image
7. Use "Save to Assets" to save your creation

**Tips for Better Results:**
- Be specific and detailed in your descriptions
- Try different style templates for varied results
- For better quality, increase the steps and guidance scale (with Standard mode)
- For faster generation, use LCM mode with fewer steps
- Use negative prompts to avoid unwanted elements

## Technical Details

Little Geeky's Learning Adventure integrates several cutting-edge technologies:

### AI Integration

The application leverages Ollama to run local LLMs, providing:
- Text generation for reading exercises
- Math problem creation
- Visual content analysis
- Educational content adaptation
- Image generation from text descriptions

Models can be configured for different tasks in the Settings tab, allowing optimization for specific educational purposes.

### Image Generation Technology

The Image Creator uses:
- **Stable Diffusion**: State-of-the-art text-to-image generation
- **LCM (Latent Consistency Model)**: For faster image generation
- **Memory Optimization**: Techniques to improve performance on limited hardware
- **Style Templates**: Pre-configured prompts for consistent artistic styles
- **Safety Development Roadmap**: Content filtering and guardrails planned for future updates

### Voice Technology

Little Geeky incorporates two text-to-speech systems:

1. **System Voices**: Using pyttsx3 to access operating system voices
2. **Kokoro TTS**: High-quality, efficient neural TTS with multiple voices

The voice system includes:
- Voice caching for improved performance
- Automatic SSML tag handling
- Cross-platform compatibility
- Efficient audio processing with numpy

### UI Framework

Built with Gradio, the application features:
- Responsive design that works on various devices
- Accessible interface elements
- Tab-based navigation for intuitive use
- Unified styling system with theme support

### Educational Design

The application incorporates research-backed educational approaches:
- Age-appropriate content tailored to developmental levels
- Support for multiple learning styles (visual, auditory, kinesthetic)
- Growth mindset encouragement
- Positive reinforcement systems
- Creative expression through image generation

## Future Development

Little Geeky's Learning Adventure is an ongoing project with plans for future enhancements:

- **Content Safety Filters**: Implementation of NSFW filtering for the Image Creator (high priority)
- **Child-Safe Prompt Templates**: Pre-vetted prompt templates designed specifically for educational content
- **Additional subject areas**: Science, Social Studies and more
- **More interactive exercises and games**: Enhanced gamification elements
- **Expanded accessibility features**: Making education inclusive for all learners
- **Animation capabilities**: Bringing generated images to life with simple animations
- **Integration with additional educational resources**: Expanding the knowledge base

## Licensing

This project is released under the MIT License, which allows for free use, modification, and distribution with minimal restrictions. For the complete terms and conditions, please refer to the LICENSE file included in this repository.
Important Note: While this project itself uses the MIT License, it incorporates various third-party dependencies and models that may be governed by different licensing terms. Users are responsible for reviewing and complying with all applicable licenses for each component used.

Please consult the documentation of individual dependencies and models to ensure compliance with their specific licensing requirements.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

---

Little Geeky's Learning Adventure - Making learning an adventure, one skill at a time.
