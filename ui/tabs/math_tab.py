# ui/tabs/math_tab.py
import gradio as gr
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
                """Generate age-appropriate math problems based on educational research"""
                try:
                    # Get model from settings
                    model = SettingsManager.get_model_for_task("text")
                    logger.info(f"Using text model from settings: {model}")
                    
                    # Define research-based, grade-appropriate math problem types
                    # Each group has unified guidelines appropriate for the entire grade range
                    level_specifics = {
                        "Grade 1-2": {
                            "addition": {
                                "prompt": "Create a word problem about adding within 20.",
                                "constraints": "Use numbers between 1-20. Focus on concrete objects that can be visualized.",
                                "examples": "Counting toys, combining groups of animals, adding items in a shopping basket."
                            },
                            "subtraction": {
                                "prompt": "Create a math problem about subtracting within 20.",
                                "constraints": "Use numbers between 1-20. Use 'take away' scenarios or 'how many more' comparison problems.",
                                "examples": "Sharing treats, losing items, comparing collections."
                            },
                            "multiplication": {
                                "prompt": "Create a very simple multiplication problem as repeated addition.",
                                "constraints": "Only use small numbers 1-5, and present it as adding the same number multiple times.",
                                "examples": "Adding the same number of stickers to several pages, placing the same number of toys in each box."
                            },
                            "division": {
                                "prompt": "Create a very simple division problem as sharing equally.",
                                "constraints": "Only use small numbers that divide evenly with no remainders. Total should be 20 or less.",
                                "examples": "Sharing cookies equally among friends, putting the same number of stickers on each page."
                            },
                            "counting": {
                                "prompt": "Create a problem about counting objects or skip-counting.",
                                "constraints": "Count by 1s, 2s, 5s, or 10s up to 100. Use concrete, countable objects.",
                                "examples": "Counting stars in groups, arranging objects in equal groups, filling containers."
                            }
                        },
                        "Grade 3-4": {
                            "addition": {
                                "prompt": "Create a multi-digit addition word problem.",
                                "constraints": "Use 2-3 digit numbers. May include simple regrouping (carrying).",
                                "examples": "Adding prices, combining measurements, finding total distances."
                            },
                            "subtraction": {
                                "prompt": "Create a multi-digit subtraction word problem.",
                                "constraints": "Use 2-3 digit numbers. May include simple regrouping (borrowing).",
                                "examples": "Finding differences in scores, calculating change, determining how much more is needed."
                            },
                            "multiplication": {
                                "prompt": "Create a word problem using multiplication.",
                                "constraints": "Use multiplication facts up to 12×12. Emphasize equal groups, arrays, or area models.",
                                "examples": "Arranging seats in rows, calculating total wheels on vehicles, finding the area of a rectangle."
                            },
                            "division": {
                                "prompt": "Create a division problem with a clear context.",
                                "constraints": "Use division with quotients up to 12 with no remainders or simple remainders.",
                                "examples": "Sharing items equally, organizing into equal groups, determining how many groups can be made."
                            },
                            "fractions": {
                                "prompt": "Create a problem involving simple fractions.",
                                "constraints": "Use unit fractions and basic fraction comparisons. Focus on part-whole relationships.",
                                "examples": "Sharing a pizza, dividing a chocolate bar, measuring ingredients for a recipe."
                            },
                            "measurement": {
                                "prompt": "Create a problem about measuring length, weight, time, or volume.",
                                "constraints": "Include unit conversions within the same system (inches to feet, minutes to hours).",
                                "examples": "Measuring ingredients, determining elapsed time, comparing weights of objects."
                            }
                        },
                        "Grade 5-6": {
                            "addition": {
                                "prompt": "Create a word problem involving addition of decimals or fractions.",
                                "constraints": "Use decimals to the hundredths place or fractions with different denominators.",
                                "examples": "Adding measurements with precision, combining ingredients in recipes, calculating total costs."
                            },
                            "subtraction": {
                                "prompt": "Create a word problem involving subtraction of decimals or fractions.",
                                "constraints": "Use decimals to the hundredths place or fractions with different denominators.",
                                "examples": "Finding differences in measurements, calculating change, comparing amounts."
                            },
                            "multiplication": {
                                "prompt": "Create a multi-digit multiplication word problem.",
                                "constraints": "Include 2-3 digit by 1-2 digit multiplication. May involve decimals.",
                                "examples": "Calculating costs for multiple items, finding areas of larger rectangles, scaling recipes."
                            },
                            "division": {
                                "prompt": "Create a long division problem with a real-world context.",
                                "constraints": "Division with 2-3 digit dividends and 1-2 digit divisors. May include remainders or decimals.",
                                "examples": "Sharing costs, finding averages, determining rates or unit prices."
                            },
                            "decimal": {
                                "prompt": "Create a problem with decimal operations.",
                                "constraints": "Use decimals to the hundredths place. Connect to money or measurement contexts.",
                                "examples": "Calculating total costs, measuring with precision, comparing metric measurements."
                            },
                            "percentage": {
                                "prompt": "Create a percentage calculation problem.",
                                "constraints": "Use percentages from 1% to 100%. Include discount, tax, or proportion scenarios.",
                                "examples": "Finding discounts, calculating tax amounts, determining percentage changes."
                            },
                            "ratio": {
                                "prompt": "Create a problem involving ratios or proportions.",
                                "constraints": "Use ratios and proportional relationships in real-world contexts.",
                                "examples": "Scaling recipes, mixing paint colors, determining speeds or rates."
                            }
                        }
                    }
                    
                    # First, check if the topic explicitly mentions a specific operation
                    explicit_operation = None
                    if "addition" in topic.lower() or "add" in topic.lower():
                        explicit_operation = "addition"
                    elif "subtraction" in topic.lower() or "subtract" in topic.lower():
                        explicit_operation = "subtraction"
                    elif "multiplication" in topic.lower() or "multiply" in topic.lower():
                        explicit_operation = "multiplication"
                    elif "division" in topic.lower() or "divide" in topic.lower():
                        explicit_operation = "division"
                        
                    # If explicit operation found, make sure it's available for the selected grade level
                    if explicit_operation and explicit_operation in level_specifics[level]:
                        matched_type = explicit_operation
                    else:
                        # If no explicit operation or not available for grade level, use the matching logic
                        import random
                        problem_types = list(level_specifics.get(level, level_specifics["Grade 1-2"]).keys())
                        
                        # Map user-entered topics to appropriate problem types
                        operation_keywords = {
                            "addition": ["plus", "sum", "combine", "total", "altogether"],
                            "subtraction": ["minus", "difference", "take away", "remove", "left"],
                            "multiplication": ["times", "product", "repeated addition", "groups of"],
                            "division": ["quotient", "split", "share", "groups", "equal parts"],
                            "fractions": ["fraction", "part", "half", "third", "fourth", "quarter"],
                            "decimal": ["decimal", "tenths", "hundredths", "point"],
                            "percentage": ["percent", "%", "discount", "interest", "portion"],
                            "ratio": ["ratio", "proportion", "comparative", "scale", "relationship"],
                            "measurement": ["measure", "meter", "inch", "pound", "kilogram", "time", "hour"],
                            "counting": ["count", "skip-count", "how many", "tally"]
                        }
                        
                        # Try to match the topic to a problem type
                        matched_type = None
                        for ptype, keywords in operation_keywords.items():
                            if any(keyword in topic.lower() for keyword in keywords):
                                # Check if this problem type is appropriate for the grade level
                                if ptype in problem_types:
                                    matched_type = ptype
                                    break
                        
                        # If no match or inappropriate match, pick randomly from grade-appropriate types
                        if not matched_type:
                            matched_type = random.choice(problem_types)
                    
                    # Get the specific information for this problem type
                    problem_info = level_specifics[level][matched_type]
                    
                    # Log what type of problem we're generating
                    logger.info(f"Generating {matched_type} problem for {level}")
                    
                    # Create a detailed prompt for the model that incorporates educational best practices
                    prompt = f"""Create a {level} math problem about {topic}, using ONLY the operation: {matched_type}

PROBLEM TYPE: {problem_info['prompt']}

EDUCATIONAL CONSTRAINTS: 
{problem_info['constraints']}

EXAMPLES OF CONTEXTS:
{problem_info['examples']}

IMPORTANT REQUIREMENTS:
1. You MUST create a {matched_type} problem only - do not use other operations
2. Make the problem age-appropriate and engaging for the entire {level} range
3. Use clear, simple language with a meaningful context
4. Avoid common names (like Tommy or Emily) and stereotypical scenarios
5. Include visual elements that can be imagined (spatial thinking aids learning)
6. Use culturally diverse contexts and inclusive scenarios

Format the result exactly like this:
Problem: [the complete word problem]
Answer: [just the numerical answer, including units if applicable]

Remember to make the problem creative, realistic, and focused on conceptual understanding!"""
                    
                    response = await self.app.ollama.generate_response_async(model, prompt)
                    
                    try:
                        parts = response.split("Answer:")
                        problem = parts[0].replace("Problem:", "").strip()
                        answer = parts[1].strip() if len(parts) > 1 else ""
                        
                        self.correct_answer = extract_number(answer)
                        
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
                            audio_file = self.app.audio.text_to_speech_async(problem, voice_id, speed)
                            self.last_problem_audio = audio_file
                        
                        return problem, audio_file, "", None, None

                    except Exception as e:
                        logger.error(f"Error parsing math problem: {e}")
                        return response, None, "", None, None
                except Exception as e:
                    logger.error(f"Error generating math problem: {e}")
                    return f"Error generating math problem: {str(e)}", None, "", None, None

            async def check_answer(answer, voice, speed):
                try:
                    if not answer:
                        result = "Please provide an answer! 📝"
                    else:
                        user_num = extract_number(answer)
                        
                        if user_num is None:
                            result = "Please provide a numerical answer!"
                        elif self.correct_answer is None:
                            result = "Sorry, I couldn't verify the answer. Please try another problem!"
                        else:
                            is_correct = abs(float(user_num) - float(self.correct_answer)) < 0.01
                            if is_correct:
                                result = "🌟 Correct! Great job! 🌟"
                                if self.app.current_user:
                                    progress = UserProgress(self.app.current_user)
                                    progress.update_stat("math_solved", 1)
                                    newly_earned = await self.app.achievement_manager.check_achievements(progress)
                                    if newly_earned:
                                        result += "\n\n🎉 You also earned new achievements! Check the Achievements tab!"
                            else:
                                result = f"Not quite right. The correct answer is {self.correct_answer}. Try again! 💪"
                    
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
                        result_audio_file = self.app.audio.text_to_speech_async(result, voice_id, speed)
                    
                    return result, self.last_problem_audio, result_audio_file
                    
                except Exception as e:
                    logger.error(f"Error checking answer: {e}")
                    return "There was a problem checking your answer. Please try again!", self.last_problem_audio, None

            def update_calculator(button_value, current_display):
                if button_value == "C":
                    return "0"
                elif button_value == "⌫":
                    return current_display[:-1] if len(current_display) > 1 else "0"
                elif button_value == "=":
                    try:
                        expression = current_display.replace("×", "*").replace("÷", "/")
                        result = safe_eval(expression)
                        return str(result)
                    except:
                        return "Error"
                else:
                    if current_display == "0" and button_value not in "÷×+-()":
                        return button_value
                    return current_display + button_value

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