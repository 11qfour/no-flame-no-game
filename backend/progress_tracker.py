"""
Модуль для отслеживания прогресса обработки видео
"""

class ProgressTracker:
    def __init__(self):
        self.current_progress = 0
        self.current_message = ""
    
    def update_progress(self, progress: float, message: str = ""):
        """Обновляет текущий прогресс"""
        self.current_progress = progress
        if message:
            self.current_message = message
        print(f"⏳ Прогресс: {progress:.1f}% - {message}")
    
    def get_progress(self):
        """Возвращает текущий прогресс"""
        return {
            "progress": self.current_progress,
            "message": self.current_message
        }
    
    def reset(self):
        """Сбрасывает прогресс"""
        self.current_progress = 0
        self.current_message = ""

# Глобальный экземпляр трекера прогресса
progress_tracker = ProgressTracker()
