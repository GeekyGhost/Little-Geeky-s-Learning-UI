# Little Geeky's Learning Adventure

<img width="772" alt="reading" src="https://github.com/user-attachments/assets/d956b3f9-8a38-493d-85bb-9f667493c854" />

<img width="388" alt="math" src="https://github.com/user-attachments/assets/d70aa3e4-fc06-4671-8fa8-f8233b32ac35" />

<img width="577" alt="typing" src="https://github.com/user-attachments/assets/de10815e-1cb6-4b9f-864a-a6d0a0ff27e1" />

## About

Little Geeky's Learning Adventure is an interactive educational platform designed to make learning engaging and fun for elementary school children (grades 1-6). This application combines cutting-edge AI technologies with research-backed educational methodologies to create a supportive, adaptive learning environment.

Developed as a passion project by a single developer, Little Geeky represents countless hours of development, research, and testing to create a tool that can help children develop crucial skills in reading, mathematics, and typing.

## Features

- **AI-Powered Learning**: Leverages local large language models (via Ollama) to provide personalized, contextually relevant educational content
- **Multi-Modal Learning**: Supports text, audio, and image-based learning experiences
- **Voice Interaction**: High-quality text-to-speech capabilities bring content to life
- **Achievement System**: Motivates learners with achievement tracking and progress visualization
- **Customizable Experience**: Settings for voice, speed, theme, and more
- **Educational Focus Areas**: Reading comprehension, mathematics, and typing skills

## Installation

### Prerequisites

- **Python 3.8+**: Required for the application's core functionality
- **Ollama**: To run local LLMs (Large Language Models)
- **Windows, macOS, or Linux**: Supported operating systems

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

#### 5. Directory Structure

After installation, these directories will be created:

- `data/` - Stores user progress and settings
- `logs/` - Contains application logs for troubleshooting
- `models/` - Used for caching model information
- `temp/` - Temporary files for document processing

### System Requirements

- **Minimum**:
  - CPU: Dual-core processor
  - RAM: 4GB (8GB recommended if using large AI models)
  - Storage: 500MB for the application + space for models (models can range from 1GB to 8GB each)
  - Network: Not required after installation
  
- **Recommended** (for optimal performance):
  - CPU: Quad-core processor
  - RAM: 16GB or more
  - GPU: NVIDIA GPU with CUDA support (for faster AI processing)
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

- **Document processing issues**:
  - Ensure PyMuPDF is properly installed
  - Check if document is password protected
  - Verify file format is supported (PDF, PNG, JPG)

- **For log file access**:
  - Look in `logs/little_geeky.log` for detailed error messages

## Interface & Navigation

Little Geeky features a tab-based interface with the following main areas:

- **Reading Tab**: For reading practice, document analysis, and comprehension
- **Math Tab**: For mathematics exercises tailored to different grade levels
- **Typing Tab**: For keyboard skills development and practice
- **Achievements Tab**: For tracking learning progress and accomplishments
- **Settings Tab**: For customizing the learning experience

## Detailed Usage Guide

### Reading Tab

The Reading Tab offers a comprehensive environment for children to practice and improve their reading skills.

**Features:**
- **Document Upload**: Support for PDFs and images
- **Text Processing**: Extract and format text from uploaded documents
- **AI Analysis**: Generate summaries, explanations, or ask questions about texts
- **Text-to-Speech**: Have text read aloud with adjustable voices and speeds
- **Voice Recording**: Use your microphone to interact with the application

**How to Use:**
1. Upload documents using the file uploader or paste text directly
2. Navigate between pages of uploaded documents using the Previous/Next buttons
3. Type your request (e.g., "summarize this", "explain what this means") in the request field
4. Click "Process Request" to generate AI responses
5. Click "Read Aloud" to hear the text spoken with your selected voice

### Math Tab

The Math Tab provides age-appropriate mathematics exercises and tools to develop numerical skills.

**Features:**
- **Personalized Math Problems**: Generate problems appropriate for grades 1-6
- **Interactive Calculator**: Built-in calculator for practice and verification
- **Voice Instructions**: Audio explanations of math problems
- **Adaptive Difficulty**: Select different grade ranges (1-2, 3-4, 5-6)
- **AI-Generated Word Problems**: Engaging, context-rich math scenarios

**How to Use:**
1. Select a type of math problem (addition, subtraction, multiplication, division)
2. Choose your grade level
3. Click "Get Problem!" to generate a new math exercise
4. Use the calculator if needed
5. Type your answer in the solution area
6. Click "Check Answer" to verify your solution

### Typing Tab

The Typing Tab helps children develop their keyboard skills through engaging exercises.

**Features:**
- **Custom Typing Exercises**: Generate exercises based on interests and topics
- **Difficulty Levels**: Easy, Medium, and Hard options
- **Accuracy Tracking**: Detailed feedback on typing performance
- **Voice Instructions**: Audio guidance for typing tasks
- **Progress Tracking**: Monitor improvement over time

**How to Use:**
1. Enter a topic of interest (e.g., "dinosaurs", "space", "animals")
2. Select a difficulty level
3. Click "Get Exercise!" to generate a typing challenge
4. Type the displayed text in the input area
5. Click "Check My Typing!" to receive feedback on accuracy and speed

### Achievements Tab

The Achievements Tab visualizes learning progress and encourages continued engagement.

**Features:**
- **Achievement Badges**: Visual representations of accomplishments
- **Progress Statistics**: Numerical tracking of completed exercises
- **Learning Milestones**: Recognition of significant learning moments
- **Daily Streaks**: Tracking of consistent usage

**How to Use:**
1. Login to track your progress
2. View your earned achievements and locked achievements
3. Check your statistics for each learning area
4. Click "Refresh Achievements" to update after completing exercises

### Settings Tab

The Settings Tab allows customization of the application experience.

**Features:**
- **Global Settings**: Voice, speed, theme, and font size adjustments
- **Model Management**: Configure AI models for different tasks
- **Voice Customization**: Select from system voices or Kokoro TTS voices
- **Theme Selection**: Choose from various visual themes including accessibility options
- **Achievement Configuration**: Customize achievement parameters

**How to Use:**
1. Adjust global settings to your preferences
2. Manage AI models for different tasks
3. Configure achievement settings
4. Apply changes and refresh settings as needed

## Technical Details

Little Geeky's Learning Adventure integrates several cutting-edge technologies:

### AI Integration

The application leverages Ollama to run local LLMs, providing:
- Text generation for reading exercises
- Math problem creation
- Visual content analysis
- Educational content adaptation

Models can be configured for different tasks in the Settings tab, allowing optimization for specific educational purposes.

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

## Architecture Highlights

Little Geeky's Learning Adventure demonstrates several noteworthy design patterns and techniques:

### Modular Structure

The application is organized into discrete components:
- **UI Module**: Presentation and interaction (tabs, buttons, forms)
- **Services Module**: External interactions (TTS, LLM, document processing)
- **Models Module**: Data management (user progress, achievements)
- **Utils Module**: Helper functions (text processing, file handling)

### Performance Optimizations

Several techniques improve application performance:
- Voice data caching system for faster TTS
- Efficient text chunking for large documents
- Optimized audio processing pipeline
- Lazy loading of resources

### Error Resilience

The application includes robust error handling:
- Graceful fallbacks for unavailable services
- Clear user feedback for issues
- Logging system for troubleshooting
- Resource cleanup to prevent leaks

## Acknowledgements

Little Geeky's Learning Adventure would not be possible without these amazing open-source projects:

- **[Gradio](https://gradio.app/)**: For the powerful, easy-to-use UI framework
- **[Kokoro TTS](https://github.com/mrtian/kokoro-tts)**: For the high-quality text-to-speech capabilities
- **[Ollama](https://ollama.ai/)**: For enabling local LLM access
- **[PyMuPDF](https://pymupdf.readthedocs.io/)**: For PDF processing capabilities
- **[PyTorch](https://pytorch.org/)**: For the underlying ML infrastructure
- **[pyttsx3](https://github.com/nateshmbhat/pyttsx3)**: For system voice access
- **[SpeechRecognition](https://github.com/Uberi/speech_recognition)**: For voice input processing

Special thanks to the broader open-source community and educational researchers whose work has informed the pedagogical approaches used in this application.

## Future Development

Little Geeky's Learning Adventure is an ongoing project with plans for future enhancements:

- Additional subject areas (Science, Social Studies)
- More interactive exercises and games
- Expanded accessibility features
- Integration with additional educational resources

## Licensing

This project is released under the MIT License, which allows for free use, modification, and distribution with minimal restrictions. For the complete terms and conditions, please refer to the LICENSE file included in this repository.
Important Note: While this project itself uses the MIT License, it incorporates various third-party dependencies and models that may be governed by different licensing terms. Users are responsible for reviewing and complying with all applicable licenses for each component used.

Please consult the documentation of individual dependencies and models to ensure compliance with their specific licensing requirements.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository.

---

Little Geeky's Learning Adventure - Making learning an adventure, one skill at a time.
