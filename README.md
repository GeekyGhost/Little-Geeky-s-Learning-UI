# Little Geeky's Learning Adventure

<img width="600" alt="reading" src="https://github.com/user-attachments/assets/d956b3f9-8a38-493d-85bb-9f667493c854" />
<img width="600" alt="math" src="https://github.com/user-attachments/assets/d70aa3e4-fc06-4671-8fa8-f8233b32ac35" />
<img width="600" alt="typing" src="https://github.com/user-attachments/assets/de10815e-1cb6-4b9f-864a-a6d0a0ff27e1" />
<img width="600" alt="image-creator" src="https://github.com/user-attachments/assets/851621d3-caae-4aff-ba1e-398fbb3095a1" />

> ⚠️ **IMPORTANT SAFETY NOTICE** ⚠️
> 
> **PARENTAL/ADULT SUPERVISION IS REQUIRED**  
> 
> Little Geeky's Learning Adventure is designed as an **attended learning tool**. Adults should actively monitor and engage with children during its use. This is particularly important for features like the Image Creator, which currently does not implement NSFW filtering.
>
> The platform is meant to be a collaborative learning experience between adults and children, not a substitute for adult guidance and supervision.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Educational Approach](#educational-approach)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Windows Installation](#windows-installation)
  - [macOS & Linux Installation](#macos--linux-installation)
  - [GPU Acceleration Setup](#gpu-acceleration-setup)
- [Usage Guide](#usage-guide)
  - [Reading Tab](#reading-tab)
  - [Math Tab](#math-tab)
  - [Typing Tab](#typing-tab)
  - [Image Creator Tab](#image-creator-tab)
  - [Achievements Tab](#achievements-tab)
  - [Settings Tab](#settings-tab)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)
- [Technical Details](#technical-details)
- [License](#license)

## About

Little Geeky's Learning Adventure is an interactive educational platform designed to make learning engaging and fun for elementary school children (grades 1-6). This application combines cutting-edge AI technologies with research-backed educational methodologies to create a supportive, adaptive learning environment.

Developed as a passion project by a single developer, Little Geeky represents countless hours of development, research, and testing to create a tool that can help children develop crucial skills in reading, mathematics, typing, and creative expression through image generation.

**Note:** This application is currently optimized for Windows users, though it can run on macOS and Linux with additional configuration.

## Features

- **AI-Powered Learning**: Leverages local large language models (via Ollama) to provide personalized, contextually relevant educational content
- **Multi-Modal Learning**: Supports text, audio, image-based learning experiences
- **Voice Interaction**: High-quality text-to-speech capabilities bring content to life
- **Achievement System**: Motivates learners with achievement tracking and progress visualization
- **Customizable Experience**: Settings for voice, speed, theme, and more
- **Educational Focus Areas**: Reading comprehension, mathematics, typing skills, and creative image creation
- **Offline Functionality**: Works without internet connection after initial setup

## Educational Approach

Little Geeky's Learning Adventure is built upon several well-established educational methodologies and principles:

### 1. Personalized Learning

The platform adapts to each child's abilities and interests through AI-powered content generation. This personalization helps maintain engagement and provides an appropriate level of challenge for each learner, following Vygotsky's Zone of Proximal Development theory.

### 2. Multi-Modal Learning

By engaging multiple senses through text, audio, and visual elements, the platform caters to different learning styles (visual, auditory, kinesthetic). Research indicates that multi-modal approaches enhance comprehension and retention by creating diverse neural pathways for information processing.

### 3. Growth Mindset Development

The achievement system is designed to reward effort and progress rather than just final outcomes, encouraging a growth mindset as described by Carol Dweck. Children learn that abilities can be developed through dedication and hard work.

### 4. Scaffolded Learning

Each activity provides appropriate support and guidance, gradually fading as the child demonstrates mastery. This scaffolded approach helps learners build confidence and independence over time.

### 5. Inquiry-Based Learning

Many activities encourage exploration and discovery, allowing children to construct their own understanding rather than passively receiving information. This constructivist approach promotes deeper learning and critical thinking skills.

### 6. Immediate Feedback

The platform provides instant feedback on activities, helping children understand their progress and make necessary adjustments. This feedback loop accelerates learning and reduces frustration.

### 7. Creative Expression

Through the Image Creator and upcoming Book Maker, children can express their ideas visually and narratively, developing creative thinking and communication skills.

### 8. Parental Involvement

The platform is designed for collaborative use, encouraging parents to engage with their children's learning, ask questions, and extend activities beyond the screen - an approach supported by extensive research on parental involvement in education.

## Installation

### Prerequisites

- **Windows 10/11**: Primary supported platform (64-bit)
- **Python 3.8+**: Required for the application's core functionality
- **Ollama**: To run local LLMs (Large Language Models)
- **Stable Diffusion Models**: Required for the Image Creator feature
- **Minimum Hardware**:
  - CPU: Dual-core processor
  - RAM: 8GB minimum (16GB recommended)
  - Storage: 1GB for the application + space for models (models range from 1GB to 8GB each)
  - GPU: Optional but recommended for faster image generation

### Windows Installation

1. **Install Python 3.8 or newer**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure you check "Add Python to PATH" during installation

2. **Install Ollama**
   - Download and install from [Ollama's official website](https://ollama.ai/download)
   - Once installed, keep Ollama running in the background while using Little Geeky

3. **Clone or Download Little Geeky**
   ```
   git clone https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git
   cd Little-Geeky-s-Learning-UI
   ```
   Alternatively, download the ZIP file from GitHub and extract it

4. **Run the Installation Script**
   - Double-click `run.bat` or run it from the command line
   - This script will create a virtual environment and install all dependencies

5. **Download Required Models in Ollama**
   - Open Command Prompt and run:
     ```
     ollama pull llama3
     ollama pull llava
     ```

6. **Set Up Image Creator Models**
   - Create a `Checkpoints` folder in the application directory if it doesn't already exist
   - Download the GeekyGhost LCM model from [Civitai](https://civitai.com/models/476202/geekyghost-lcm)
   - Place the downloaded `.safetensors` file in the `Checkpoints` folder

7. **Launch the Application**
   - Run `python main.py` from the command line in the application directory or
   - Double-click `run.bat` again to launch the application

### macOS & Linux Installation

1. **Install Python 3.8 or newer**
   - macOS: Use Homebrew: `brew install python3`
   - Linux: Use your distribution's package manager, e.g., `sudo apt install python3 python3-pip`

2. **Install Ollama**
   - Download and install from [Ollama's website](https://ollama.ai/download)
   - Follow the platform-specific instructions

3. **Clone the Repository**
   ```
   git clone https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git
   cd Little-Geeky-s-Learning-UI
   ```

4. **Run the Installation Script**
   ```
   chmod +x run.sh
   ./run.sh
   ```

5. **Download Required Models**
   - Same as Windows instructions above

6. **Additional Requirements for Linux**
   - Install additional audio libraries: `sudo apt-get install libsndfile1`

**Note**: While the application can run on macOS and Linux, it has been primarily tested and optimized for Windows. Some features may require additional configuration on other platforms.

### GPU Acceleration Setup

GPU acceleration can significantly improve performance, especially for the Image Creator feature.

#### For NVIDIA GPUs:

1. **Install the latest NVIDIA drivers** from the [official website](https://www.nvidia.com/Download/index.aspx)

2. **Install CUDA Toolkit**
   - Download CUDA 11.8 or 12.1 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation wizard (Express Installation recommended)

3. **Install PyTorch with CUDA support**
   ```
   # For CUDA 11.8
   pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify Installation**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU device name: {torch.cuda.get_device_name(0)}")
   ```

#### For AMD GPUs:

AMD GPU support is limited for the deep learning libraries used in this application. We recommend using CPU mode for now.

#### For Apple Silicon (M1/M2/M3):

1. **Install PyTorch with MPS support**
   ```
   pip install torch torchvision
   ```

2. **Verify installation**
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

## Usage Guide

### First Launch

When launching Little Geeky for the first time:

1. The application will create necessary directories for storing user data and models
2. Enter a username to track progress and achievements
3. Navigate through the tabs to access different learning activities

### Reading Tab

The Reading Tab focuses on reading comprehension and document analysis:

- **Document Upload**: Upload PDFs or images for reading practice
- **Text Analysis**: AI assistance in understanding complex text
- **Reading Exercises**: Practice comprehension with AI-generated questions
- **Voice Narration**: Text-to-speech capabilities for reading aloud

**Parent Guidance**: Help your child select appropriate reading materials. Use the AI-generated questions as conversation starters. Encourage your child to summarize what they've read in their own words.

### Math Tab

The Math Tab provides adaptive mathematics exercises:

- **Customizable Difficulty**: Adjust math problems based on grade level and ability
- **Problem Generation**: AI creates diverse math problems
- **Visual Aids**: Visual representations of math concepts
- **Interactive Calculator**: Built-in calculator for checking work

**Parent Guidance**: Begin with problems slightly below your child's current level to build confidence, then gradually increase difficulty. Ask your child to explain their problem-solving approach rather than focusing solely on the correct answer.

### Typing Tab

The Typing Tab helps develop keyboard skills:

- **Typing Exercises**: Progressive typing challenges
- **Speed & Accuracy Tracking**: Monitors typing performance
- **Custom Content**: Create personalized typing exercises
- **Ergonomic Tips**: Guidance on proper hand positioning

**Parent Guidance**: Ensure proper posture and hand positioning. Start with short, frequent sessions rather than long practice periods. Emphasize accuracy over speed initially.

### Image Creator Tab

> ⚠️ **SAFETY NOTICE**: Adult supervision required. No content filtering system is currently implemented.

The Image Creator allows children to transform text descriptions into images:

- **Text-to-Image**: Generate images from descriptions
- **Style Templates**: Quick selection of artistic styles
- **Customization Options**: Adjust image parameters
- **Transparent Backgrounds**: Create images with transparent backgrounds

**Parent Guidance**: Directly supervise this feature at all times. Help frame appropriate prompts and discuss the generated images. Use this as an opportunity to talk about digital creativity and how AI interprets text.

### Achievements Tab

The Achievements Tab tracks learning progress:

- **Progress Visualization**: Visual representation of learning journey
- **Skill Tracking**: Monitors development across different areas
- **Achievement Unlocking**: Milestone recognition to motivate learning
- **Statistics**: Detailed stats on usage and improvement

**Parent Guidance**: Celebrate achievements together, focusing on effort rather than just outcomes. Use the statistics to identify strengths and areas that might need more attention.

### Settings Tab

The Settings Tab allows customization of the learning experience:

- **Voice Selection**: Choose from multiple voice options
- **Theme Customization**: Adjust visual appearance
- **Model Management**: Configure AI models for different tasks
- **Accessibility Options**: Adjust text size and speed

**Parent Guidance**: Involve your child in customizing their learning environment. This provides a sense of ownership and can increase engagement.

## Troubleshooting

### Application Doesn't Start

- Verify Python is correctly installed: `python --version`
- Ensure the virtual environment is activated
- Check logs in the `logs/` directory for specific errors

### No AI Responses

- Verify Ollama is running in the background
- Check that models are downloaded in Ollama
- Configure models in the Settings tab

### No Audio Output

- Check system audio settings
- Ensure audio libraries are installed
- Try switching to system voices in the settings

### Image Creator Doesn't Work

- Verify you have models in the `Checkpoints` folder
- Check for GPU memory issues if you have a GPU
- Try lowering image dimensions or steps in the UI

### Performance Issues

- Close other resource-intensive applications
- Reduce model complexity in Settings
- Consider upgrading hardware for optimal experience

For detailed error information, check the log file at `logs/little_geeky.log`.

## Future Development

Little Geeky's Learning Adventure is an ongoing project with several planned enhancements:

### Coming Soon

- **Content Safety Filters**: Implementation of NSFW filtering for the Image Creator
- **Programming Tab**: Introduction to coding concepts with integrated Phaser game engine
- **Little Geeky Book Maker**: A new tab allowing children to create illustrated short books using the Image Generator and LLM
- **Child-Safe Prompt Templates**: Pre-vetted prompt templates designed specifically for educational content

### Future Roadmap

- **Additional Subject Areas**: Science, Social Studies, and more
- **More Interactive Exercises**: Enhanced gamification elements
- **Expanded Accessibility Features**: Making education inclusive for all learners
- **Animation Capabilities**: Bringing generated images to life with simple animations
- **Integration with Additional Educational Resources**: Expanding the knowledge base
- **Cross-Platform Optimization**: Better support for macOS and Linux

## Technical Details

Little Geeky's Learning Adventure integrates several cutting-edge technologies:

### AI Integration

The application leverages Ollama to run local LLMs, providing:
- Text generation for reading exercises
- Math problem creation
- Visual content analysis
- Educational content adaptation
- Image generation from text descriptions

### Image Generation Technology

The Image Creator uses:
- **Stable Diffusion**: State-of-the-art text-to-image generation
- **LCM (Latent Consistency Model)**: For faster image generation
- **Memory Optimization**: Techniques to improve performance on limited hardware

### Voice Technology

Little Geeky incorporates two text-to-speech systems:
1. **System Voices**: Using pyttsx3 to access operating system voices
2. **Kokoro TTS**: High-quality, efficient neural TTS with multiple voices

### UI Framework

Built with Gradio, the application features:
- Responsive design that works on various devices
- Accessible interface elements
- Tab-based navigation for intuitive use
- Unified styling system with theme support

## License

This project is released under the MIT License, which allows for free use, modification, and distribution with minimal restrictions. For the complete terms and conditions, please refer to the LICENSE file included in this repository.

**Important Note**: While this project itself uses the MIT License, it incorporates various third-party dependencies and models that may be governed by different licensing terms. Users are responsible for reviewing and complying with all applicable licenses for each component used.

---

<p align="center">
  <b>Little Geeky's Learning Adventure</b><br>
  <i>Making learning an adventure, one skill at a time.</i>
</p>
