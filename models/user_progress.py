# models/user_progress.py
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Set, Any, Optional
from utils.logging_utils import logger
from config.settings import USERS_DIR

@dataclass
class UserProgress:
    """Tracks user progress and achievements"""
    user_id: str
    achievements: Set[str] = None
    stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.achievements is None:
            self.achievements = set()
        if self.stats is None:
            self.stats = {
                "reading_completed": 0,
                "math_solved": 0,
                "typing_exercises": 0,
                "typing_accuracy": [],
                "last_login": None,
                "streak_days": 0,
                "documents_processed": 0,
                "pdfs_read": 0,
                "images_analyzed": 0,
                "recent_activities": []
            }
        self.load_progress()
        self._update_streak()
    
    def load_progress(self):
        """Load user progress from file"""
        try:
            os.makedirs(USERS_DIR, exist_ok=True)
            try:
                with open(f'{USERS_DIR}/{self.user_id}.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.achievements = set(data.get('achievements', []))
                    self.stats = data.get('stats', self.stats)
            except FileNotFoundError:
                self.save_progress()
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    
    def save_progress(self):
        """Save user progress to file"""
        try:
            os.makedirs(USERS_DIR, exist_ok=True)
            with open(f'{USERS_DIR}/{self.user_id}.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'achievements': list(self.achievements),
                    'stats': self.stats
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _update_streak(self):
        """Update daily login streak"""
        today = datetime.now().date()
        last_login = None
        
        if self.stats["last_login"]:
            try:
                last_login = datetime.fromisoformat(self.stats["last_login"]).date()
            except (ValueError, TypeError):
                last_login = None
        
        if last_login:
            days_diff = (today - last_login).days
            if days_diff == 1:
                self.stats["streak_days"] += 1
            elif days_diff > 1:
                self.stats["streak_days"] = 1
        else:
            self.stats["streak_days"] = 1
            
        self.stats["last_login"] = today.isoformat()
        self.save_progress()
    
    def update_stat(self, stat_type: str, value: Any):
        """Update a user statistic"""
        if stat_type == "typing_accuracy":
            self.stats["typing_accuracy"].append(value)
        else:
            self.stats[stat_type] = self.stats.get(stat_type, 0) + value
        
        # Add to recent activities
        self.stats["recent_activities"].append({
            "timestamp": datetime.now().isoformat(),
            "type": stat_type,
            "value": value
        })
        
        # Keep only last 50 activities
        self.stats["recent_activities"] = self.stats["recent_activities"][-50:]
        self.save_progress()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of user statistics"""
        accuracy_avg = (
            sum(self.stats["typing_accuracy"]) / len(self.stats["typing_accuracy"])
            if self.stats["typing_accuracy"] else 0
        )
        return {
            "Total Exercises": {
                "Reading": self.stats["reading_completed"],
                "Math": self.stats["math_solved"],
                "Typing": self.stats.get("typing_exercises", 0)
            },
            "Typing Accuracy": f"{accuracy_avg:.1f}%",
            "Achievement Count": len(self.achievements),
            "Daily Streak": self.stats["streak_days"],
            "Last Active": self.stats["last_login"]
        }

    def update_document_stats(self, doc_type: str):
        """Update document reading statistics"""
        if doc_type == "pdf":
            self.stats["pdfs_read"] = self.stats.get("pdfs_read", 0) + 1
        elif doc_type == "image":
            self.stats["images_analyzed"] = self.stats.get("images_analyzed", 0) + 1
        self.stats["documents_processed"] = self.stats.get("documents_processed", 0) + 1
        self.save_progress()