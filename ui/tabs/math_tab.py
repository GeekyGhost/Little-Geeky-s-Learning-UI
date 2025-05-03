# ui/tabs/math_tab.py
import gradio as gr
import random
from utils.logging_utils import logger
from utils.math_utils import safe_eval
from utils.text_utils import extract_number
from models.user_progress import UserProgress
from utils.settings_utils import SettingsManager

class MathTab:
    def __init__(self, app_context):
        self.app = app_context
        self.last_problem_audio = None
        self.correct_answer = None
        self.problem_context = None  # Store additional problem context for debugging
        
    def generate_problem_data(self, operation, grade_level):
        """Generate random numbers and correct answer based on operation and grade level"""
        import random
        
        # Define grade-appropriate number ranges
        if grade_level == "Grade 1-2":
            if operation == "addition":
                a = random.randint(1, 10)
                b = random.randint(1, 10)
                answer = a + b
                return {"numbers": [a, b], "answer": answer, "operation": "+"}
            elif operation == "subtraction":
                b = random.randint(1, 5)
                a = random.randint(b+1, 10)  # Ensure a > b to avoid negative numbers
                answer = a - b
                return {"numbers": [a, b], "answer": answer, "operation": "-"}
            elif operation == "multiplication":
                a = random.randint(1, 5)
                b = random.randint(1, 5)
                answer = a * b
                return {"numbers": [a, b], "answer": answer, "operation": "×"}
            elif operation == "division":
                b = random.randint(1, 5)
                a = b * random.randint(1, 4)  # Ensure clean division
                answer = a // b
                return {"numbers": [a, b], "answer": answer, "operation": "÷"}
                
        elif grade_level == "Grade 3-4":
            if operation == "addition":
                a = random.randint(10, 50)
                b = random.randint(10, 50)
                answer = a + b
                return {"numbers": [a, b], "answer": answer, "operation": "+"}
            elif operation == "subtraction":
                b = random.randint(10, 30)
                a = random.randint(b+1, 80)  # Ensure a > b
                answer = a - b
                return {"numbers": [a, b], "answer": answer, "operation": "-"}
            elif operation == "multiplication":
                a = random.randint(2, 10)
                b = random.randint(2, 10)
                answer = a * b
                return {"numbers": [a, b], "answer": answer, "operation": "×"}
            elif operation == "division":
                b = random.randint(2, 10)
                a = b * random.randint(1, 10)  # Ensure clean division
                answer = a // b
                return {"numbers": [a, b], "answer": answer, "operation": "÷"}
                
        elif grade_level == "Grade 5-6":
            if operation == "addition":
                a = random.randint(50, 500)
                b = random.randint(50, 500)
                answer = a + b
                return {"numbers": [a, b], "answer": answer, "operation": "+"}
            elif operation == "subtraction":
                b = random.randint(50, 200)
                a = random.randint(b+1, 700)  # Ensure a > b
                answer = a - b
                return {"numbers": [a, b], "answer": answer, "operation": "-"}
            elif operation == "multiplication":
                a = random.randint(5, 20)
                b = random.randint(5, 15)
                answer = a * b
                return {"numbers": [a, b], "answer": answer, "operation": "×"}
            elif operation == "division":
                b = random.randint(2, 12)
                a = b * random.randint(2, 15)  # Ensure clean division
                answer = a // b
                return {"numbers": [a, b], "answer": answer, "operation": "÷"}
            elif operation == "decimal":
                # Simplified decimal operations
                a = round(random.randint(100, 500) / 100, 2)  # Two decimal places
                b = round(random.randint(100, 500) / 100, 2)
                operation_type = random.choice(["+", "-"])
                if operation_type == "+":
                    answer = round(a + b, 2)
                    return {"numbers": [a, b], "answer": answer, "operation": "+"}
                else:
                    if a < b:  # Swap to avoid negative
                        a, b = b, a
                    answer = round(a - b, 2)
                    return {"numbers": [a, b], "answer": answer, "operation": "-"}
            elif operation == "percentage":
                # Simplified percentage problems
                whole = random.randint(20, 100)
                percentage = random.choice([10, 20, 25, 50, 75])
                answer = whole * percentage / 100
                answer = int(answer) if answer.is_integer() else round(answer, 1)
                return {"numbers": [whole, percentage], "answer": answer, "operation": "%"}
        
        # Default fallback with simple addition
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return {"numbers": [a, b], "answer": a + b, "operation": "+"}

    def check_answer_math(self, user_answer, correct_answer):
        """Check if the user's answer is mathematically correct"""
        try:
            # Handle empty answers
            if not user_answer or user_answer.strip() == "":
                return False
                
            # Clean user input of non-numeric characters except decimal point
            cleaned_answer = ''.join(c for c in user_answer.replace('$', '').replace(',', '').strip() 
                                    if c.isdigit() or c == '.' or c == '-')
            
            # Handle cases where the user didn't provide a valid number
            if not cleaned_answer or cleaned_answer == '.' or cleaned_answer == '-':
                return False
            
            # Convert to appropriate numeric type
            if isinstance(correct_answer, int) or correct_answer.is_integer():
                user_value = int(float(cleaned_answer))  # Handle cases where user types "5.0" for 5
                correct_value = int(correct_answer)
                return user_value == correct_value
            else:
                user_value = float(cleaned_answer)
                correct_value = float(correct_answer)
                # Allow for small rounding differences
                return abs(user_value - correct_value) < 0.01
        except Exception as e:
            logger.error(f"Error checking answer: {e}")
            return False
    
    def create_tab(self, model_dropdown=None) -> gr.Tab:
        # Get model info for display
        models_info_html = self._get_models_info_html()
        
        with gr.Tab("Math 🔢") as tab:
            with gr.Row():
                with gr.Column(scale=2):
                    problem_input = gr.Textbox(
                        lines=2,
                        label="What kind of math problem would you like?",
                        placeholder="E.g., addition, multiplication, word problems...",
                        elem_classes="math-input"
                    )
                    
                    with gr.Row():
                        gr.Button(
                            "Addition ➕",
                            elem_classes="math-type-btn",
                            variant="secondary"
                        ).click(
                            fn=lambda: "addition problems",
                            outputs=[problem_input]
                        )
                        gr.Button(
                            "Subtraction ➖",
                            elem_classes="math-type-btn",
                            variant="secondary"
                        ).click(
                            fn=lambda: "subtraction problems",
                            outputs=[problem_input]
                        )
                        gr.Button(
                            "Multiplication ✖️",
                            elem_classes="math-type-btn",
                            variant="secondary"
                        ).click(
                            fn=lambda: "multiplication problems",
                            outputs=[problem_input]
                        )
                        gr.Button(
                            "Division ➗",
                            elem_classes="math-type-btn",
                            variant="secondary"
                        ).click(
                            fn=lambda: "division problems",
                            outputs=[problem_input]
                        )
                    
                    with gr.Row():
                        level = gr.Radio(
                            ["Grade 1-2", "Grade 3-4", "Grade 5-6"],
                            label="Grade Level",
                            value="Grade 1-2"
                        )
                        
                        # Get voice settings
                        voice_settings = SettingsManager.get_voice_settings()
                        default_voice = voice_settings.get("voice", "System Default")
                        
                        # Get list of voice names
                        voice_names = [v.name for v in self.app.audio.voices] if self.app.audio.voices else ["Default System Voice"]
                        if default_voice not in voice_names and voice_names:
                            default_voice = voice_names[0]
                            
                        # Add debug logging to check voices
                        logger.info(f"Math tab voice choices: {voice_names}")
                        
                        voice_dropdown = gr.Dropdown(
                            choices=voice_names,
                            label="Choose Voice",
                            value=default_voice
                        )
                        send_btn = gr.Button("Get Problem! 🔢", variant="primary")
                        
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
                    
                    with gr.Column(elem_classes="calculator") as calc:
                        display = gr.Textbox(
                            label="Calculator",
                            value="0",
                            interactive=False,
                            elem_classes="calc-display"
                        )
                        
                        calc_buttons = []
                        button_layout = [
                            ["C", "⌫", "(", ")"],
                            ["7", "8", "9", "÷"],
                            ["4", "5", "6", "×"],
                            ["1", "2", "3", "-"],
                            ["0", ".", "=", "+"]
                        ]
                        
                        for row in button_layout:
                            with gr.Row():
                                for btn_text in row:
                                    btn_class = (
                                        "function-btn" if btn_text in ["C", "⌫", "(", ")"]
                                        else "operator-btn" if btn_text in ["÷", "×", "-", "+", "="]
                                        else "number-btn"
                                    )
                                    calc_buttons.append(
                                        gr.Button(
                                            btn_text,
                                            elem_classes=f"calc-btn {btn_class}"
                                        )
                                    )

            problem_area = gr.Textbox(
                label="Problem",
                interactive=False,
                lines=4,
                elem_classes="math-problem"
            )
            
            # Set autoplay from settings
            autoplay = voice_settings.get("autoplay", True)
            problem_audio = gr.Audio(
                label="Listen to Problem",
                elem_classes="audio-player",
                autoplay=autoplay
            )
            
            solution_area = gr.Textbox(
                label="Your Solution",
                placeholder="Type your answer here...",
                elem_classes="math-solution"
            )
            
            check_btn = gr.Button("Check Answer ✅")
            result_area = gr.Textbox(
                label="Results",
                interactive=False,
                elem_classes="result-box"
            )
            result_audio = gr.Audio(
                label="Listen to Result",
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

            async def get_math_problem(topic, level, voice, speed):
                """Generate age-appropriate math problems with Python and LLM contextualization"""
                try:
                    # Get model from settings
                    model = SettingsManager.get_model_for_task("text")
                    logger.info(f"Using text model from settings: {model}")
                    
                    # Determine which operation to use based on topic
                    operation = "addition"  # Default
                    if "add" in topic.lower() or "sum" in topic.lower() or "plus" in topic.lower():
                        operation = "addition"
                    elif "subtract" in topic.lower() or "minus" in topic.lower() or "difference" in topic.lower():
                        operation = "subtraction"
                    elif "multiply" in topic.lower() or "times" in topic.lower() or "product" in topic.lower():
                        operation = "multiplication"
                    elif "divide" in topic.lower() or "quotient" in topic.lower() or "sharing" in topic.lower():
                        operation = "division"
                    elif "decimal" in topic.lower():
                        operation = "decimal"
                    elif "percent" in topic.lower() or "%" in topic.lower():
                        operation = "percentage"
                    
                    # Generate problem data with Python
                    problem_data = self.generate_problem_data(operation, level)
                    
                    # Store correct answer in class variable for checking later
                    self.correct_answer = problem_data["answer"]
                    
                    # Store problem context for debugging
                    self.problem_context = {
                        "operation": operation,
                        "level": level,
                        "numbers": problem_data["numbers"],
                        "expected_answer": problem_data["answer"]
                    }
                    logger.info(f"Generated problem data: {self.problem_context}")
                    
                    # Use LLM to create a word problem around these numbers with much clearer instructions
                    prompt = f"""Create a {level} math word problem about {topic} that uses ONLY the operation {operation}.

The problem must use EXACTLY these numbers: {problem_data['numbers']}
Each number should be used EXACTLY ONCE in the calculation.
The answer must be: {problem_data['answer']}

VERY IMPORTANT RULES:
1. Do NOT create problems where you need to use the same number multiple times
2. For example, if given [7, 7], do not create a problem requiring 7+7+7+7
3. Only use the exact operation requested: {operation}
4. The problem should require EXACTLY ONE step to solve 
5. Make sure the problem is solvable by a direct calculation with the given numbers

Format the result exactly like this:
Problem: [the complete word problem]"""

                    response = await self.app.ollama.generate_response_async(model, prompt)
                    
                    # Extract just the problem text
                    problem = response.replace("Problem:", "").strip()
                    
                    # Find voice ID and generate audio
                    voice_id = None
                    for v in self.app.audio.voices:
                        if v.name == voice:
                            voice_id = v.id
                            logger.info(f"Found voice ID {voice_id} for voice {voice}")
                            break
                    
                    if not voice_id and self.app.audio.voices:
                        voice_id = self.app.audio.voices[0].id
                        
                    audio_file = None
                    if voice_id:
                        audio_file = self.app.audio.text_to_speech_async(problem, voice_id, speed)
                        self.last_problem_audio = audio_file
                    
                    return problem, audio_file, "", None, None

                except Exception as e:
                    logger.error(f"Error generating math problem: {e}")
                    return f"Error generating math problem: {str(e)}", None, "", None, None

            async def check_answer(answer, voice, speed):
                """Check if the user's answer is correct using Python verification"""
                try:
                    if not answer:
                        result = "Please provide an answer! 📝"
                    else:
                        # Log the answer checking process for debugging
                        logger.info(f"Checking answer - User input: '{answer}', Expected answer: {self.correct_answer}")
                        
                        # Use Python to check the answer
                        is_correct = self.check_answer_math(answer, self.correct_answer)
                        
                        # Also log the context for more thorough debugging
                        if hasattr(self, 'problem_context') and self.problem_context:
                            logger.info(f"Problem context: {self.problem_context}")
                        
                        if is_correct:
                            result = "🌟 Correct! Great job! 🌟"
                            if self.app.current_user:
                                progress = UserProgress(self.app.current_user)
                                progress.update_stat("math_solved", 1)
                                newly_earned = await self.app.achievement_manager.check_achievements(progress)
                                if newly_earned:
                                    result += "\n\n🎉 You also earned new achievements! Check the Achievements tab!"
                        else:
                            # Include the operation in the feedback to help with learning
                            operation_symbol = ""
                            if hasattr(self, 'problem_context') and self.problem_context:
                                op = self.problem_context.get("operation", "")
                                if op == "addition": operation_symbol = "+"
                                elif op == "subtraction": operation_symbol = "-"
                                elif op == "multiplication": operation_symbol = "×"
                                elif op == "division": operation_symbol = "÷"
                                elif op == "percentage": operation_symbol = "%"
                            
                            # Format the result based on whether it's an integer or float
                            if isinstance(self.correct_answer, (int, float)) and self.correct_answer.is_integer():
                                formatted_answer = int(self.correct_answer)
                            else:
                                formatted_answer = self.correct_answer
                                
                            result = f"Not quite right. The correct answer is {formatted_answer}. Try again! 💪"
                    
                    # Find voice ID from voice name
                    voice_id = None
                    for v in self.app.audio.voices:
                        if v.name == voice:
                            voice_id = v.id
                            logger.info(f"Found voice ID {voice_id} for voice {voice}")
                            break
                    
                    if not voice_id and self.app.audio.voices:
                        voice_id = self.app.audio.voices[0].id
                            
                    result_audio_file = None
                    if voice_id:
                        result_audio_file = self.app.audio.text_to_speech_async(result, voice_id, speed)
                    
                    return result, self.last_problem_audio, result_audio_file
                    
                except Exception as e:
                    logger.error(f"Error checking answer: {e}")
                    return "There was a problem checking your answer. Please try again!", self.last_problem_audio, None

            def update_calculator(button_value, current_display):
                try:
                    if button_value == "C":
                        return "0"
                    elif button_value == "⌫":
                        return current_display[:-1] if len(current_display) > 1 else "0"
                    elif button_value == "=":
                        try:
                            expression = current_display.replace("×", "*").replace("÷", "/")
                            result = safe_eval(expression)
                            # Format result for better display
                            if isinstance(result, float) and result.is_integer():
                                return str(int(result))
                            return str(result)
                        except Exception as e:
                            logger.error(f"Calculator error: {e}")
                            return "Error"
                    else:
                        if current_display == "0" and button_value not in "÷×+-()":
                            return button_value
                        return current_display + button_value
                except Exception as e:
                    logger.error(f"Calculator error: {e}")
                    return "Error"

            send_btn.click(
                fn=get_math_problem,
                inputs=[problem_input, level, voice_dropdown, speed_slider],
                outputs=[problem_area, problem_audio, result_area, result_audio, solution_area]
            )

            check_btn.click(
                fn=check_answer,
                inputs=[solution_area, voice_dropdown, speed_slider],
                outputs=[result_area, problem_audio, result_audio]
            )

            for btn in calc_buttons:
                btn.click(
                    fn=update_calculator,
                    inputs=[btn, display],
                    outputs=[display]
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
