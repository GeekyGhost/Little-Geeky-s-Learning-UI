# ui/tabs/typing_tab.py
import gradio as gr
from models.user_progress import UserProgress
from utils.logging_utils import logger
from utils.settings_utils import SettingsManager

class TypingTab:
    def __init__(self, app_context):
        self.app = app_context
        
    def create_tab(self, model_dropdown=None) -> gr.Tab:
        # Get model info for display
        models_info_html = self._get_models_info_html()
        
        with gr.Tab("Typing Practice ⌨️") as tab:
            with gr.Row():
                with gr.Column(scale=2):
                    topic_input = gr.Textbox(
                        lines=2,
                        label="What topic would you like to practice typing about?",
                        placeholder="E.g., animals, space, dinosaurs..."
                    )
                    
                    with gr.Row():
                        difficulty = gr.Radio(
                            ["Easy", "Medium", "Hard"],
                            label="Difficulty Level",
                            value="Easy"
                        )
                        
                        # Get voice settings
                        voice_settings = SettingsManager.get_voice_settings()
                        default_voice = voice_settings.get("voice", "System Default")
                        
                        # Get list of voice names
                        voice_names = [v.name for v in self.app.audio.voices] if self.app.audio.voices else ["Default System Voice"]
                        if default_voice not in voice_names and voice_names:
                            default_voice = voice_names[0]
                        
                        # Add debug logging to check voices
                        logger.info(f"Typing tab voice choices: {voice_names}")
                        
                        voice_dropdown = gr.Dropdown(
                            choices=voice_names,
                            label="Choose Voice",
                            value=default_voice
                        )
                        send_btn = gr.Button("Get Exercise! ⌨️", variant="primary")
                        
                with gr.Column(scale=1):
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        stats_val = progress.get_stats_summary()
                    else:
                        stats_val = {"Total Exercises": {"Reading": 0, "Math": 0, "Typing": 0},
                                     "Typing Accuracy": "0.0%",
                                     "Achievement Count": 0,
                                     "Daily Streak": 0,
                                     "Last Active": None}
                    stats = gr.JSON(
                        value=stats_val,
                        label="Your Progress",
                        every=1
                    )

                    # Use speed from settings
                    default_speed = voice_settings.get("speed", 1.0)
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=default_speed,
                        step=0.1,
                        label="Reading Speed"
                    )
                    
                    # Display which models are being used
                    models_info = gr.HTML(
                        value=models_info_html,
                        label="Selected Models",
                        elem_classes="models-info"
                    )
            
            exercise_area = gr.Textbox(
                label="Type this:",
                interactive=False,
                lines=6,
                elem_classes="typing-exercise"
            )
            
            # Set autoplay from settings
            autoplay = voice_settings.get("autoplay", True)
            instructions_audio = gr.Audio(
                label="Listen to Instructions",
                elem_classes="audio-player",
                autoplay=autoplay
            )
            
            practice_area = gr.Textbox(
                lines=6,
                label="Type here:",
                placeholder="Type the text exactly as shown above...",
                elem_classes="typing-input"
            )
            
            check_btn = gr.Button("Check My Typing! ✅", variant="primary")
            result_area = gr.Textbox(
                label="Results",
                interactive=False,
                elem_classes="result-box"
            )
            result_audio = gr.Audio(
                label="Listen to Results",
                elem_classes="audio-player",
                autoplay=autoplay
            )
            
            # Add autoplay JavaScript helper
            gr.HTML("""
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
            """, visible=False)

            async def get_typing_exercise(topic, difficulty, voice, speed):
                """Get typing exercise - use model from settings"""
                try:
                    # Get model from settings
                    model = SettingsManager.get_model_for_task("text")
                    logger.info(f"Using text model from settings: {model}")
                    
                    difficulty_prompts = {
                        "Easy": {
                            "prompt": """Create a single simple sentence (10-15 words) about {topic}.
                            Use basic words and simple punctuation. The sentence should be fun and educational.
                            Only return the sentence to type, no other text.""",
                            "instructions": "This is an easy typing exercise. Type one simple sentence, including the punctuation and capital letters exactly as shown."
                        },
                        "Medium": {
                            "prompt": """Create a short paragraph (3-4 sentences) about {topic}.
                            Use moderate vocabulary and basic punctuation. Make it educational and engaging.
                            Only return the paragraph to type, no other text.""",
                            "instructions": "This is a medium difficulty exercise. Type the entire paragraph, paying attention to sentence breaks and punctuation."
                        },
                        "Hard": {
                            "prompt": """Create two medium-length paragraphs (4-5 sentences each) about {topic}.
                            Use more complex vocabulary and proper punctuation. Make it challenging and educational.
                            Only return the paragraphs to type, no other text.""",
                            "instructions": "This is a challenging exercise with two paragraphs. Type everything exactly as shown, including paragraph breaks, punctuation, and capital letters."
                        }
                    }
                    
                    prompt = difficulty_prompts[difficulty]["prompt"].format(topic=topic)
                    response = await self.app.ollama.generate_response_async(model, prompt)
                    
                    # Generate audio instructions
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
                        instructions = (
                            f"Here's your {difficulty.lower()} typing exercise about {topic}. "
                            f"{difficulty_prompts[difficulty]['instructions']} "
                            "Take your time and focus on accuracy rather than speed. "
                            "When you're done, click the Check My Typing button to see how you did."
                        )
                        audio_file = self.app.audio.text_to_speech_async(instructions, voice_id, speed)
                    
                    clean_response = response.replace("Exercise:", "").replace("Type this:", "").strip()
                    
                    # Increment typing exercise count
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        progress.update_stat("typing_exercises", 1)
                        await self.app.achievement_manager.check_achievements(progress)
                        stats_data = progress.get_stats_summary()
                    else:
                        stats_data = {"Correct Words": 0, "Mistakes": 0, "Accuracy": "0%"}
                    
                    return clean_response, "", stats_data, audio_file, None
                except Exception as e:
                    logger.error(f"Error generating typing exercise: {e}")
                    return "Error generating typing exercise. Please try again.", "", {"Error": str(e)}, None, None

            async def check_typing(original, typed, voice, speed):
                if not typed:
                    result = "Please type something!"
                    stats = {"Correct Words": 0, "Mistakes": 0, "Accuracy": "0%"}
                    accuracy = 0
                else:
                    original_words = original.split()
                    typed_words = typed.split()
                    
                    correct_words = sum(1 for orig, typ in zip(original_words, typed_words) if orig == typ)
                    total_words = len(original_words)
                    mistakes = len([1 for orig, typ in zip(original_words, typed_words) if orig != typ])
                    
                    if len(typed_words) < len(original_words):
                        mistakes += len(original_words) - len(typed_words)
                    elif len(typed_words) > len(original_words):
                        mistakes += len(typed_words) - len(original_words)
                    
                    accuracy = (correct_words / total_words * 100) if total_words > 0 else 0
                    
                    if self.app.current_user:
                        progress = UserProgress(self.app.current_user)
                        progress.update_stat("typing_accuracy", accuracy)
                        await self.app.achievement_manager.check_achievements(progress)
                        stats = progress.get_stats_summary()
                    else:
                        stats = {
                            "Correct Words": correct_words,
                            "Mistakes": mistakes,
                            "Accuracy": f"{accuracy:.1f}%"
                        }
                    
                    if accuracy >= 95:
                        result = "🌟 Outstanding job! Your typing is excellent! 🌟"
                    elif accuracy >= 85:
                        result = "👏 Great work! Keep practicing to get even better! 👏"
                    elif accuracy >= 70:
                        result = "👍 Good effort! Try to type each word carefully! 👍"
                    else:
                        result = "💪 Keep practicing! Take your time and focus on accuracy! 💪"
                    
                    if mistakes > 0:
                        result += "\n\nWatch out for: "
                        if len(typed_words) != len(original_words):
                            result += "missing or extra words, "
                        result += "capitalization, spelling, and punctuation!"
                
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
                        
                result_audio_file = None
                if voice_id:
                    audio_feedback = (
                        f"{result} You typed {correct_words} words correctly "
                        f"and made {mistakes} mistakes. "
                        f"Your accuracy is {accuracy:.1f}%."
                    )
                    result_audio_file = self.app.audio.text_to_speech_async(audio_feedback, voice_id, speed)
                
                return result, stats, result_audio_file

            send_btn.click(
                fn=get_typing_exercise,
                inputs=[topic_input, difficulty, voice_dropdown, speed_slider],
                outputs=[exercise_area, practice_area, stats, instructions_audio, result_audio]
            )

            check_btn.click(
                fn=check_typing,
                inputs=[exercise_area, practice_area, voice_dropdown, speed_slider],
                outputs=[result_area, stats, result_audio]
            )

            return tab
    
    def _get_models_info_html(self):
        """Generate HTML displaying which models are being used"""
        try:
            text_model = SettingsManager.get_model_for_task("text")
            
            return f"""
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 0.9em;">
                <p><strong>Text Model:</strong> {text_model}</p>
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