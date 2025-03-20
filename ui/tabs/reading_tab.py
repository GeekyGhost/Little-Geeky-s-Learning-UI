# ui/tabs/reading_tab.py
import gradio as gr
import asyncio
import re
from utils.logging_utils import logger
from models.user_progress import UserProgress
from services.document_service import DocumentProcessor
from utils.text_utils import format_text_for_speech
from utils.settings_utils import SettingsManager

class ReadingTab:
    def __init__(self, app_context):
        self.app = app_context
        self.uploaded_documents = []
        self.current_document_index = 0
    
    def create_tab(self, model_dropdown=None) -> gr.Tab:
        # Initialize models info HTML at the beginning
        models_info_html = self._get_models_info_html()
        
        with gr.Tab("Reading 📚") as tab:
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Upload Documents or Images",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        file_count="multiple",
                        elem_classes="file-upload"
                    )
                    
                    text_input = gr.Textbox(
                        lines=4,
                        label="What would you like me to read?",
                        placeholder="Type, paste text, or upload files above...",
                        elem_classes="reading-input"
                    )
                    
                    request_input = gr.Textbox(
                        lines=2,
                        label="Enter your request",
                        placeholder="e.g., 'summarize this text', 'explain the main points'...",
                        elem_classes="request-input"
                    )
                    
                    with gr.Row():
                        # Get global voice settings
                        voice_settings = SettingsManager.get_voice_settings()
                        default_voice = voice_settings.get("voice", "System Default")
                        
                        # Get list of voice names
                        voice_names = [v.name for v in self.app.audio.voices] if self.app.audio.voices else ["Default System Voice"]
                        if default_voice not in voice_names and voice_names:
                            default_voice = voice_names[0]
                        
                        # Add debug logging to check voices
                        logger.info(f"Reading tab voice choices: {voice_names}")
                        
                        voice_dropdown = gr.Dropdown(
                            choices=voice_names,
                            label="Choose a Voice",
                            value=default_voice,
                            elem_classes="voice-selector"
                        )
                    
                    with gr.Row():
                        prev_btn = gr.Button("◀️ Previous", variant="secondary")
                        next_btn = gr.Button("Next ▶️", variant="secondary")
                    
                    with gr.Row():
                        record_btn = gr.Button("Record Voice 🎤", variant="secondary")
                        process_btn = gr.Button("Process Request 💭", variant="primary")
                        read_btn = gr.Button("Read Aloud 📢", variant="primary")
                        
                with gr.Column(scale=1):
                    image_preview = gr.Image(
                        label="Preview",
                        show_label=False,
                        height=400,
                        elem_classes="preview-image"
                    )
                    
                    # Use speed from global settings
                    default_speed = voice_settings.get("speed", 1.0)
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=default_speed,
                        step=0.1,
                        label="Reading Speed"
                    )
                    
                    stats = gr.JSON(
                        value={"Reading Progress": "Login to track progress"},
                        label="Progress",
                        elem_classes="reading-stats"
                    )
                    
                    doc_counter = gr.HTML(
                        value="Document: 0/0",
                        label="Document Counter",
                        elem_classes="doc-counter"
                    )
                    
                    # Display which models are being used for this tab
                    models_info = gr.HTML(
                        value=models_info_html,  # Use pre-generated HTML
                        label="Selected Models",
                        elem_classes="models-info" 
                    )

            response_area = gr.HTML(
                value="",
                label="Response",
                elem_classes="reading-response"
            )
            
            # Set autoplay based on settings
            autoplay = voice_settings.get("autoplay", True)
            audio_player = gr.Audio(
                label="Listen",
                elem_classes="audio-player",
                autoplay=autoplay
            )

            # Ensure the necessary JS is added to enable autoplay
            autoplay_js = """
            <script>
                // Add a Mutation Observer to watch for audio elements being added to the DOM
                const audioObserver = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.addedNodes) {
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeName === 'AUDIO' || (node.querySelector && node.querySelector('audio'))) {
                                    const audioElem = node.nodeName === 'AUDIO' ? node : node.querySelector('audio');
                                    if (audioElem) {
                                        setTimeout(() => {
                                            audioElem.play().catch(e => console.log('Auto-play prevented:', e));
                                        }, 300);
                                    }
                                }
                            });
                        }
                    });
                });
                
                // Start observing audio elements in the app
                setTimeout(() => {
                    const targetNode = document.querySelector('.gradio-container');
                    if (targetNode) {
                        audioObserver.observe(targetNode, { childList: true, subtree: true });
                    }
                }, 1000);
            </script>
            """
            
            gr.HTML(autoplay_js, visible=False)

            # Define handlers
            async def process_uploads(files):
                if not files:
                    return "", "Please upload files first! 📝", None, "Document: 0/0", {"Reading Progress": "Login to track progress"}

                # Clean up any existing temporary files
                if self.uploaded_documents:
                    DocumentProcessor.cleanup_temp_files(self.uploaded_documents)
                
                self.uploaded_documents = []
                processed_content = ""
                preview_path = None
                
                if self.app.current_user:
                    progress = UserProgress(self.app.current_user)
                else:
                    progress = None

                # Get model settings before processing
                vision_model = SettingsManager.get_model_for_task("vision")
                logger.info(f"Using vision model from settings: {vision_model}")
                
                for file in files:
                    try:
                        file_path = file.name
                        # Process document and get pages
                        pages = self.app.document_processor.process_document(file_path)
                        
                        for page in pages:
                            if page['type'] == 'image':
                                # Get image description using the vision model from settings
                                description = await DocumentProcessor.describe_image(
                                    page['path'], 
                                    self.app.ollama, 
                                    vision_model  # Use the vision model from settings
                                )
                                page['content'] = description
                                if progress:
                                    progress.update_document_stats("image")
                            elif page['type'] == 'pdf_page':
                                if progress:
                                    progress.update_document_stats("pdf")

                            self.uploaded_documents.append(page)
                    
                    except Exception as e:
                        logger.error(f"Error processing file: {e}")
                
                if self.uploaded_documents:
                    self.current_document_index = 0
                    current_doc = self.uploaded_documents[0]
                    processed_content = current_doc['content']
                    preview_path = current_doc['path']
                    
                    if progress:
                        progress.update_stat("documents_processed", len(self.uploaded_documents))
                        progress.update_stat("reading_completed", 1)  # Count document processing as a reading exercise
                        await self.app.achievement_manager.check_achievements(progress)
                        current_stats = progress.get_stats_summary()
                    else:
                        current_stats = {"Reading Progress": "Login to track progress"}
                    
                    doc_count = f"Document: 1/{len(self.uploaded_documents)}"
                    status_msg = f"Files processed successfully! 📄 Page {current_doc['page_number']} of {current_doc['total_pages']}"
                    return processed_content, status_msg, preview_path, doc_count, current_stats
                
                return "", "No files were processed successfully 😕", None, "Document: 0/0", {"Reading Progress": "Login to track progress"}

            def navigate_documents(direction):
                if not self.uploaded_documents:
                    return "", None, "Document: 0/0", "No documents loaded! 📝"
                
                if direction == "next":
                    self.current_document_index = (self.current_document_index + 1) % len(self.uploaded_documents)
                else:
                    self.current_document_index = (self.current_document_index - 1) % len(self.uploaded_documents)
                
                current_doc = self.uploaded_documents[self.current_document_index]
                preview_path = current_doc['path']
                doc_count = f"Document: {self.current_document_index + 1}/{len(self.uploaded_documents)}"
                status_msg = f"Viewing page {current_doc['page_number']} of {current_doc['total_pages']}"
                
                return current_doc['content'], preview_path, doc_count, status_msg

            async def process_as_request(request, context, voice, speed, selected_model=None):
                try:
                    # If no model is provided, use the text model from settings
                    if not selected_model:
                        selected_model = SettingsManager.get_model_for_task("text")
                        logger.info(f"Using text model from settings: {selected_model}")
                        # Update models info display when using default
                        models_info.update(value=self._get_models_info_html())
                        
                    if context.strip():
                        prompt = f"""Based on the following content:

{context}

Request: {request}

Please provide a clear, well-structured response."""
                    else:
                        prompt = request

                    response = await self.app.ollama.generate_response_async(
                        selected_model,
                        prompt,
                        "You are a friendly reading tutor. Please respond with clear, well-structured text."
                    )
                    
                    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
                    formatted_text = ""
                    word_counter = 0
                    
                    for paragraph in paragraphs:
                        words = paragraph.split()
                        highlighted_paragraph = ""
                        for word in words:
                            highlighted_paragraph += f'<span class="word" id="word-{word_counter}">{word}</span> '
                            word_counter += 1
                        formatted_text += f'<p>{highlighted_paragraph}</p>'
                    
                    # Find voice ID from voice name
                    voice_id = None
                    for v in self.app.audio.voices:
                        if v.name == voice:
                            voice_id = v.id
                            logger.info(f"Found voice ID {voice_id} for voice {voice}")
                            break
                    
                    if not voice_id:
                        logger.warning(f"Could not find voice ID for {voice}, using first available voice")
                        if self.app.audio.voices:
                            voice_id = self.app.audio.voices[0].id
                    
                    audio_file = None
                    if voice_id:
                        # Format the text properly for speech
                        speech_text = format_text_for_speech(response)
                        audio_file = self.app.audio.text_to_speech_async(speech_text, voice_id, speed)
                    
                    return formatted_text, audio_file
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    return "Sorry, something went wrong!", None

            async def read_text_verbatim(text, voice, speed):
                try:
                    if not text.strip():
                        return "No text to read!", None
                    
                    # Process the text to join lines that should be continuous
                    # but respect paragraph breaks and punctuation
                    
                    # First, highlight the words for display
                    words = text.split()
                    highlighted_text = ""
                    for i, word in enumerate(words):
                        highlighted_text += f'<span class="word" id="word-{i}">{word}</span> '
                    
                    # Find voice ID from voice name
                    voice_id = None
                    for v in self.app.audio.voices:
                        if v.name == voice:
                            voice_id = v.id
                            logger.info(f"Found voice ID {voice_id} for voice {voice}")
                            break
                    
                    if not voice_id:
                        logger.warning(f"Could not find voice ID for {voice}, using first available voice")
                        if self.app.audio.voices:
                            voice_id = self.app.audio.voices[0].id
                    
                    audio_file = None
                    if voice_id:
                        # Format the text properly for speech while preserving its natural flow
                        speech_text = format_text_for_speech(text)
                        audio_file = self.app.audio.text_to_speech_async(speech_text, voice_id, speed)
                    
                    return highlighted_text, audio_file
                except Exception as e:
                    logger.error(f"Error reading text: {e}")
                    return "Sorry, something went wrong!", None

            def handle_recording():
                is_recording, frames = self.app.recorder.toggle_recording()
                if frames:
                    text = self.app.recorder.process_recording(frames)
                    if text:
                        return text, "Ready to read! 👂", "Record Voice 🎤"
                    else:
                        return "", "Couldn't understand. Try again! 🤔", "Record Voice 🎤"
                
                return "", "Recording... 🎤" if is_recording else "Ready! 👂", "Stop Recording 🛑" if is_recording else "Record Voice 🎤"

            # Wire up event handlers
            file_upload.upload(
                fn=process_uploads,
                inputs=[file_upload],
                outputs=[text_input, response_area, image_preview, doc_counter, stats]
            )

            prev_btn.click(
                fn=lambda: navigate_documents("prev"),
                outputs=[text_input, image_preview, doc_counter, response_area]
            )

            next_btn.click(
                fn=lambda: navigate_documents("next"),
                outputs=[text_input, image_preview, doc_counter, response_area]
            )

            # Use text model from settings
            async def handle_process_request(request, text, voice, speed):
                model = SettingsManager.get_model_for_task("text")
                logger.info(f"Using text model from settings for request: {model}")
                return await process_as_request(request, text, voice, speed, model)
            
            process_btn.click(
                fn=handle_process_request,
                inputs=[request_input, text_input, voice_dropdown, speed_slider],
                outputs=[response_area, audio_player]
            )

            read_btn.click(
                fn=read_text_verbatim,
                inputs=[text_input, voice_dropdown, speed_slider],
                outputs=[response_area, audio_player]
            )

            record_btn.click(
                fn=handle_recording,
                outputs=[request_input, response_area, record_btn]
            )
            
            return tab
            
    def _get_models_info_html(self):
        """Generate HTML displaying which models are being used"""
        try:
            text_model = SettingsManager.get_model_for_task("text")
            vision_model = SettingsManager.get_model_for_task("vision")
            
            return f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 0.9em;">
                <p><strong>Text Model:</strong> {text_model}</p>
                <p><strong>Vision Model:</strong> {vision_model}</p>
                <p><small>Change models in Settings ⚙️ → Model Management → Task Models</small></p>
            </div>
            """
        except Exception as e:
            logger.error(f"Error generating models info: {e}")
            return """
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 0.9em;">
                <p>Error loading model information</p>
                <p><small>Check Settings ⚙️ to configure models</small></p>
            </div>
            """