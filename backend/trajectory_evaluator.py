import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import sqlite3

class TrajectoryEvaluator:
    """Система оценки качества траекторий для обучения"""
    
    def __init__(self, db_path: str = "trajectory_ratings.db"):
        # Используем абсолютный путь для базы данных
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(os.getcwd(), db_path)
        else:
            self.db_path = db_path
        
        print(f"🗄️ База данных будет создана в: {self.db_path}")
        self._init_database()
        print("⭐ TrajectoryEvaluator инициализирован")
    
    def _init_database(self):
        """Инициализация базы данных для оценок"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для оценок траекторий
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trajectory_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_filename TEXT NOT NULL,
                    trajectory_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    comment TEXT,
                    smoothness_factor REAL,
                    detection_params TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_filename, trajectory_id)
                )
            ''')
            
            # Таблица для статистики ошибок
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ База данных оценок инициализирована")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации БД: {e}")
    
    def rate_trajectory(self, video_filename: str, trajectory_id: int, 
                       rating: int, comment: str = "", 
                       smoothness_factor: float = 0.1,
                       detection_params: Dict = None) -> bool:
        """
        Оценивает траекторию
        
        Args:
            video_filename: Имя видео файла
            trajectory_id: ID траектории
            rating: Оценка 1-5 звезд
            comment: Комментарий о качестве
            smoothness_factor: Коэффициент плавности
            detection_params: Параметры детекции
            
        Returns:
            True если оценка сохранена успешно
        """
        print(f"🎯 Попытка оценки траектории: video_filename={video_filename}, trajectory_id={trajectory_id}, rating={rating}")
        
        if not (1 <= rating <= 5):
            print(f"❌ Оценка должна быть от 1 до 5, получено: {rating}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Сохраняем оценку
            cursor.execute('''
                INSERT OR REPLACE INTO trajectory_ratings 
                (video_filename, trajectory_id, rating, comment, smoothness_factor, detection_params, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_filename, 
                trajectory_id, 
                rating, 
                comment, 
                smoothness_factor,
                json.dumps(detection_params) if detection_params else None,
                datetime.now().isoformat()
            ))
            
            # Анализируем комментарий для выявления паттернов ошибок
            if comment:
                self._analyze_error_pattern(comment)
            
            conn.commit()
            conn.close()
            
            print(f"⭐ Оценка сохранена: {rating}/5 для траектории {trajectory_id}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения оценки: {e}")
            print(f"🔍 Тип ошибки: {type(e).__name__}")
            import traceback
            print(f"📋 Stack trace: {traceback.format_exc()}")
            return False
    
    def _analyze_error_pattern(self, comment: str):
        """Анализирует комментарий для выявления типов ошибок"""
        error_types = {
            'короткая': 'Слишком короткая траектория',
            'прерывается': 'Траектория прерывается',
            'неправильное направление': 'Неправильное направление движения',
            'ложное срабатывание': 'Ложное срабатывание (не человек)',
            'пропущен': 'Пропущен человек',
            'прямая': 'Слишком прямые линии',
            'неровная': 'Неровная траектория'
        }
        
        comment_lower = comment.lower()
        detected_errors = []
        
        for keyword, error_type in error_types.items():
            if keyword in comment_lower:
                detected_errors.append(error_type)
        
        # Сохраняем паттерны ошибок
        if detected_errors:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for error_type in detected_errors:
                    cursor.execute('''
                        INSERT OR REPLACE INTO error_patterns (error_type, frequency, last_seen)
                        VALUES (?, 
                                COALESCE((SELECT frequency + 1 FROM error_patterns WHERE error_type = ?), 1),
                                ?)
                    ''', (error_type, error_type, datetime.now().isoformat()))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"⚠️ Ошибка анализа паттернов: {e}")
    
    def get_trajectory_rating(self, video_filename: str, trajectory_id: int) -> Optional[Dict]:
        """Получает оценку конкретной траектории"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT rating, comment, smoothness_factor, detection_params, timestamp
                FROM trajectory_ratings
                WHERE video_filename = ? AND trajectory_id = ?
            ''', (video_filename, trajectory_id))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'rating': result[0],
                    'comment': result[1],
                    'smoothness_factor': result[2],
                    'detection_params': json.loads(result[3]) if result[3] else None,
                    'timestamp': result[4]
                }
            return None
            
        except Exception as e:
            print(f"❌ Ошибка получения оценки: {e}")
            return None
    
    def get_video_statistics(self, video_filename: str) -> Dict:
        """Получает статистику оценок для видео"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_rated,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating
                FROM trajectory_ratings
                WHERE video_filename = ?
            ''', (video_filename,))
            
            stats = cursor.fetchone()
            
            # Распределение оценок
            cursor.execute('''
                SELECT rating, COUNT(*) as count
                FROM trajectory_ratings
                WHERE video_filename = ?
                GROUP BY rating
                ORDER BY rating
            ''', (video_filename,))
            
            rating_distribution = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'total_rated': stats[0],
                'average_rating': round(stats[1], 2) if stats[1] else 0,
                'min_rating': stats[2],
                'max_rating': stats[3],
                'rating_distribution': rating_distribution
            }
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_error_patterns(self) -> List[Dict]:
        """Получает статистику типов ошибок"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT error_type, frequency, last_seen
                FROM error_patterns
                ORDER BY frequency DESC
            ''')
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'error_type': row[0],
                    'frequency': row[1],
                    'last_seen': row[2]
                })
            
            conn.close()
            return patterns
            
        except Exception as e:
            print(f"❌ Ошибка получения паттернов ошибок: {e}")
            return []
    
    def get_learning_recommendations(self) -> Dict:
        """Получает рекомендации для улучшения на основе оценок"""
        patterns = self.get_error_patterns()
        
        recommendations = {
            'common_issues': [],
            'suggested_improvements': []
        }
        
        for pattern in patterns[:5]:  # Топ-5 проблем
            recommendations['common_issues'].append({
                'type': pattern['error_type'],
                'frequency': pattern['frequency']
            })
        
        # Рекомендации на основе частых ошибок
        for pattern in patterns:
            if 'короткая' in pattern['error_type'].lower():
                recommendations['suggested_improvements'].append(
                    "Увеличить минимальную длину траектории"
                )
            elif 'прямая' in pattern['error_type'].lower():
                recommendations['suggested_improvements'].append(
                    "Уменьшить коэффициент плавности для более плавных линий"
                )
            elif 'ложное срабатывание' in pattern['error_type'].lower():
                recommendations['suggested_improvements'].append(
                    "Улучшить алгоритм детекции людей"
                )
        
        return recommendations
