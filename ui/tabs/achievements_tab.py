# ui/tabs/achievements_tab.py
# Modified to remove tab.load usage

import gradio as gr
from models.user_progress import UserProgress

class AchievementsTab:
    def __init__(self, app_context):
        self.app = app_context
        
    def create_tab(self) -> gr.Tab:
        # Pre-load achievements data if user is logged in
        initial_achievements_display = "Please login to view achievements!"
        initial_stats = {}
        
        if self.app.current_user:
            try:
                progress = UserProgress(self.app.current_user)
                # We can't await this directly here, so we'll use a refresh button instead
                initial_stats = progress.get_stats_summary()
            except Exception as e:
                logger.error(f"Error pre-loading achievement stats: {e}")
                
        with gr.Tab("Achievements 🏆") as tab:
            gr.Markdown("# Your Learning Journey 🌟")
            
            async def get_achievements_display():
                if not self.app.current_user:
                    return "Please login to view achievements!"
                
                progress = UserProgress(self.app.current_user)
                await self.app.achievement_manager.check_achievements(progress)  # Make sure achievements are up to date
                
                html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px;'>"
                
                for a in self.app.achievement_manager.achievements.values():
                    completed = a.id in progress.achievements
                    status = "🌟 Completed!" if completed else "🔒 Locked"
                    style = "background: #66CC00;" if completed else "background: #CC8800;"
                    
                    # Calculate progress toward achievement
                    progress_text = ""
                    if not completed:
                        total_exercises = (
                            progress.stats.get("reading_completed", 0) +
                            progress.stats.get("math_solved", 0) +
                            progress.stats.get("typing_exercises", 0)
                        )
                        
                        # Calculate average typing accuracy
                        typing_accuracy = 0
                        if progress.stats.get("typing_accuracy", []):
                            typing_accuracy = sum(progress.stats["typing_accuracy"]) / len(progress.stats["typing_accuracy"])
                        
                        # For each requirement, calculate progress
                        requirement_progress = []
                        for stat, required in a.requirements.items():
                            # Special handling for composite stats
                            if stat == "total_exercises":
                                current = total_exercises
                            elif stat == "typing_accuracy":
                                current = typing_accuracy
                            else:
                                current = progress.stats.get(stat, 0)
                            
                            percentage = min(100, int(current / required * 100))
                            requirement_progress.append(f"{stat}: {current}/{required} ({percentage}%)")
                        
                        if requirement_progress:
                            progress_text = "<div class='progress-details'>" + "<br>".join(requirement_progress) + "</div>"
                    
                    html += f"""
                    <div style='padding: 15px; border-radius: 10px; {style}'>
                        <h3>{a.icon} {a.name}</h3>
                        <p>{a.description}</p>
                        <p><strong>{status}</strong></p>
                        {progress_text}
                    </div>
                    """
                
                html += "</div>"
                return html

            def get_current_stats():
                if not self.app.current_user:
                    return {}
                return UserProgress(self.app.current_user).get_stats_summary()
            
            achievements_display = gr.HTML(
                value=initial_achievements_display,
                elem_classes="achievements-display"
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh Achievements 🔄")
                stats_display = gr.JSON(
                    value=initial_stats,
                    label="Overall Progress"
                )
            
            async def refresh_achievements():
                display = await get_achievements_display()
                stats = get_current_stats()
                return [display, stats]
            
            refresh_btn.click(
                fn=refresh_achievements,
                outputs=[achievements_display, stats_display]
            )
            
            # Instead of using tab.load, add an automatic refresh when shown
            # We'll have to use the refresh button as the main way to update
            
            if self.app.current_user:
                # Add a note encouraging refresh
                gr.HTML(
                    """<div style="text-align: center; margin-top: 10px;">
                        <p>Click "Refresh Achievements" to see your latest progress!</p>
                    </div>""",
                    elem_classes="refresh-note"
                )

            return tab