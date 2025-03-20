# models/achievements.py
import yaml
import dataclasses
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass
from utils.logging_utils import logger
from models.user_progress import UserProgress

@dataclass
class AchievementConfig:
    """Configuration for customizable achievements"""
    id: str
    name: str
    description: str
    icon: str
    requirements: Dict[str, Union[int, float]]
    enabled: bool = True

class AchievementManager:
    def __init__(self, config_file: str = "achievements.yaml"):
        self.config_file = Path(config_file)
        self.achievements: Dict[str, AchievementConfig] = {}
        self.load_config()

    def load_config(self):
        """Load achievement configurations"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    configs = yaml.safe_load(f)
                    if configs:
                        for config in configs:
                            self.achievements[config["id"]] = AchievementConfig(**config)
                    else:
                        self._create_default_config()
            except Exception as e:
                logger.error(f"Error loading achievement config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()

    def _create_default_config(self):
        """Create default achievement configuration"""
        self.achievements = {
            "first_login": AchievementConfig(
                id="first_login",
                name="First Steps 🎉",
                description="Login for the first time",
                icon="🎉",
                requirements={"logins": 1}
            ),
            "reading_master": AchievementConfig(
                id="reading_master",
                name="Reading Master 📚",
                description="Complete reading exercises",
                icon="📚",
                requirements={"reading_completed": 10, "documents_processed": 5}
            ),
            "math_wizard": AchievementConfig(
                id="math_wizard",
                name="Math Wizard 🔮",
                description="Solve math problems correctly",
                icon="🔮", 
                requirements={"math_solved": 10}
            ),
            "typing_pro": AchievementConfig(
                id="typing_pro",
                name="Typing Pro ⌨️",
                description="Complete typing exercises with high accuracy",
                icon="⌨️",
                requirements={"typing_exercises": 5, "typing_accuracy": 95.0}
            ),
            "daily_streak": AchievementConfig(
                id="daily_streak",
                name="Consistent Learner 📆",
                description="Login consecutive days",
                icon="📆",
                requirements={"streak_days": 5}
            ),
            "reading_explorer": AchievementConfig(
                id="reading_explorer", 
                name="Reading Explorer 📖",
                description="Read different types of documents",
                icon="📖",
                requirements={"pdfs_read": 3, "images_analyzed": 3}
            ),
            "problem_solver": AchievementConfig(
                id="problem_solver",
                name="Problem Solver 🧩",
                description="Complete various exercises",
                icon="🧩",
                requirements={"total_exercises": 50}
            )
        }
        self.save_config()

    def save_config(self):
        """Save achievement configurations to file"""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump([dataclasses.asdict(a) for a in self.achievements.values()], f)
        except Exception as e:
            logger.error(f"Error saving achievement config: {e}")

    async def check_achievements(self, user_progress: UserProgress) -> List[str]:
        """Check which achievements have been earned"""
        earned = []
        
        # Calculate total exercises for the problem_solver achievement
        total_exercises = (
            user_progress.stats.get("reading_completed", 0) +
            user_progress.stats.get("math_solved", 0) +
            user_progress.stats.get("typing_exercises", 0)
        )
        
        # Calculate average typing accuracy
        typing_accuracy = 0
        if user_progress.stats.get("typing_accuracy", []):
            typing_accuracy = sum(user_progress.stats["typing_accuracy"]) / len(user_progress.stats["typing_accuracy"])
        
        for achievement in self.achievements.values():
            if not achievement.enabled or achievement.id in user_progress.achievements:
                continue
                
            requirements_met = True
            for stat, required in achievement.requirements.items():
                # Special handling for composite stats
                if stat == "total_exercises":
                    current = total_exercises
                elif stat == "typing_accuracy":
                    current = typing_accuracy
                else:
                    current = user_progress.stats.get(stat, 0)
                
                if current < required:
                    requirements_met = False
                    break
                    
            if requirements_met:
                user_progress.achievements.add(achievement.id)
                earned.append(achievement.id)
                
        if earned:
            user_progress.save_progress()
            
        return earned