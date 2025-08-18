"""
Реальный анализатор видео без предустановок
Анализирует именно то видео, которое загрузил пользователь
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
import uuid
from backend.trajectory_smoother import TrajectorySmoother
from backend.advanced_tracker import AdvancedPersonTracker
from backend.dwell_time_analyzer import DwellTimeAnalyzer
from backend.progress_tracker import progress_tracker

class RealVideoAnalyzer:
    def __init__(self):
        """Инициализация анализатора"""
        # Параметры трекинга
        self.max_tracking_distance = 100  # Максимальное расстояние для связывания траекторий
        self.min_trajectory_length = 3    # Минимальная длина траектории для учета
        
        # Инициализируем YOLO детектор позже, когда он понадобится
        self.yolo_detector = None
        
        # Инициализируем продвинутый трекер с параметрами для YOLO
        self.advanced_tracker = AdvancedPersonTracker(
            max_disappeared=120,  # Увеличиваем - люди могут исчезать на 4 секунды (YOLO может пропускать кадры)
            min_trajectory_length=1  # Уменьшаем - учитываем даже одиночные детекции
        )
        
        # Инициализируем сглаживатель траекторий
        self.trajectory_smoother = TrajectorySmoother(smoothness_factor=0.1)
        
        # Инициализируем анализатор времени пребывания
        self.dwell_time_analyzer = DwellTimeAnalyzer(
            grid_size=50,  # Размер ячейки сетки 50x50 пикселей
            time_thresholds={
                'light': 1.0,      # 1 секунда - легкая теплота
                'medium': 3.0,     # 3 секунды - средняя теплота
                'high': 5.0,       # 5 секунд - высокая теплота
                'very_high': 10.0  # 10 секунд - очень высокая теплота
            }
        )
        
        print("🎨 TrajectorySmoother инициализирован")
        print("🚀 AdvancedPersonTracker инициализирован")
        print("🔥 DwellTimeAnalyzer инициализирован")
        
    def analyze_video(self, video_path: str) -> Dict:
        """Анализирует реальное видео"""
        self.current_video_path = video_path  # Сохраняем путь для использования в аналитике
        print(f"🎬 Открываем видео: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Не удалось открыть видео: {video_path}")
        
        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Видео: {width}x{height}, {fps} FPS, {total_frames} кадров, {duration:.1f}с")
        
        # Данные для анализа
        trajectories = {}
        people_per_frame = []
        frame_count = 0
        person_id_counter = 0
        previous_centroids = {}
        
        # Создаем уникальный ID для этого анализа
        analysis_id = str(uuid.uuid4())[:8]
        
        # Обрабатываем видео
        progress_tracker.reset()
        progress_tracker.update_progress(5, "Начало анализа")
        print("🔍 Начинаем детекцию людей...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем каждый кадр для лучшей синхронизации YOLO и трекинга
            if frame_count % 1 != 0:  # Обрабатываем каждый кадр для максимальной точности
                continue
            
            timestamp = frame_count / fps if fps > 0 else frame_count
            
            # Ленивая инициализация YOLO детектора
            if self.yolo_detector is None:
                try:
                    from backend.yolo_person_detector import YOLOPersonDetector
                    
                    # Сначала пробуем загрузить вашу кастомную модель
                    custom_model_path = "your_custom_model.pt"
                    if os.path.exists(custom_model_path):
                        print("🎯 Загружаем ВАШУ кастомную модель YOLO!")
                        self.yolo_detector = YOLOPersonDetector(model_path=custom_model_path)
                    else:
                        print("🎯 Загружаем стандартную модель YOLO")
                        self.yolo_detector = YOLOPersonDetector()
                    
                    print("✅ YOLO детектор инициализирован")
                except Exception as e:
                    print(f"⚠️ Ошибка инициализации YOLO детектора: {e}")
                    # Fallback на простую детекцию
                    self.yolo_detector = None
            
            # Детекция людей с помощью YOLO
            if self.yolo_detector:
                try:
                    detected_people = self.yolo_detector.detect_multiple_people(frame)
                    people_count = len(detected_people)
                    
                    # Преобразуем детекции в формат для трекинга
                    detections = [(p[0], p[1], p[2], p[3]) for p in detected_people]
                    
                    # Если YOLO не нашел людей, продолжаем без fallback
                    if people_count == 0:
                        print("🔄 YOLO не нашел людей в этом кадре")
                    
                except Exception as e:
                    print(f"⚠️ Ошибка YOLO детекции: {e}")
                    detected_people = []
                    people_count = 0
                    detections = []
            else:
                # YOLO детектор не инициализирован
                print("⚠️ YOLO детектор не доступен")
                detected_people = []
                people_count = 0
                detections = []
            
            # people_per_frame теперь собирается автоматически в трекере
            
            # Используем продвинутый трекер
            current_centroids = self.advanced_tracker.update(detections, frame_count)
            
            # Логируем состояние трекинга
            if frame_count % 30 == 0:  # Каждые 30 кадров
                active_trackers = len(self.advanced_tracker.trackers)
                total_detections = len(detections)
                print(f"📊 Кадр {frame_count}: детекций={total_detections}, активных трекеров={active_trackers}")
            
            # Траектории теперь собираются автоматически в трекере
            
            # Прогресс
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                progress_tracker.update_progress(progress, "Обработка кадров")
        
        cap.release()
        
        # Освобождаем ресурсы детектора
        if hasattr(self.yolo_detector, 'cleanup'):
            self.yolo_detector.cleanup()
        
        # Получаем траектории из продвинутого трекера
        filtered_trajectories = self.advanced_tracker.get_trajectories()
        people_per_frame = self.advanced_tracker.get_people_per_frame()
        
        print(f"✅ Обработка завершена. Найдено {len(filtered_trajectories)} траекторий")
        
        # Создаем визуализации
        progress_tracker.update_progress(85, "Создание визуализаций")
        visualizations = self._create_visualizations(
            filtered_trajectories, people_per_frame, 
            width, height, analysis_id, video_path, fps
        )
        
        # Генерируем аналитику
        progress_tracker.update_progress(95, "Генерация аналитики")
        analytics = self._generate_analytics(
            filtered_trajectories, people_per_frame, 
            duration, visualizations
        )
        
        # Конвертируем NumPy типы в Python типы для JSON сериализации
        progress_tracker.update_progress(100, "Завершение анализа")
        analytics = self._convert_numpy_types(analytics)
        
        return analytics
    
    def _convert_numpy_types(self, obj):
        """Конвертирует NumPy типы в Python типы для JSON сериализации"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    

    
    # Старый метод _track_people удален - теперь используем AdvancedPersonTracker
    
    def _create_visualizations(self, trajectories: Dict, people_per_frame: List,
                              width: int, height: int, analysis_id: str, 
                              video_path: str, fps: float) -> Dict:
        """Создает визуализации на основе реального видео"""
        
        # Создаем кадр-фон из видео (первый кадр)
        cap = cv2.VideoCapture(video_path)
        ret, background_frame = cap.read()
        cap.release()
        
        if not ret:
            # Если не удалось получить кадр, создаем простой фон
            background_frame = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
        
        # 1. Создаем тепловую карту с анализом времени пребывания
        heatmap_path = self._create_dwell_time_heatmap(
            trajectories, background_frame, width, height, analysis_id, fps
        )
        
        # 2. Создаем визуализацию троп
        paths_path = self._create_paths_visualization(
            trajectories, background_frame, width, height, analysis_id
        )
        
        # 3. Создаем график загруженности
        queue_path = self._create_queue_visualization(
            people_per_frame, analysis_id
        )
        
        # Добавляем информацию о времени пребывания, если анализ был успешным
        dwell_info = {}
        if hasattr(self, 'dwell_analysis_result'):
            dwell_info = {
                'dwell_time_analysis': {
                    'zones_count': len(self.dwell_analysis_result.get('zones_analysis', [])),
                    'active_zones': self.dwell_analysis_result.get('zones_analysis', [])[:5],  # Топ-5 зон
                    'grid_info': self.dwell_analysis_result.get('grid_info', {}),
                    'time_thresholds': self.dwell_time_analyzer.time_thresholds
                }
            }
        
        return {
            'heatmap': heatmap_path,
            'paths': paths_path,
            'queue': queue_path,
            **dwell_info
        }
    
    def _create_dwell_time_heatmap(self, trajectories: Dict, background: np.ndarray,
                                  width: int, height: int, analysis_id: str, fps: float) -> str:
        """Создает тепловую карту с анализом времени пребывания людей в красивом дизайне"""
        
        print(f"🔥 Создаем тепловую карту с анализом времени пребывания... FPS: {fps}")
        
        # Анализируем время пребывания
        # Проверяем, что fps валидный
        if fps <= 0:
            print(f"⚠️ Некорректный FPS: {fps}, используем значение по умолчанию 30")
            fps = 30.0
        
        dwell_analysis = self.dwell_time_analyzer.analyze_dwell_times(trajectories, fps)
        
        if 'error' in dwell_analysis:
            print(f"⚠️ Ошибка анализа времени пребывания: {dwell_analysis['error']}")
            # Fallback на простую тепловую карту
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Создаем красивую тепловую карту в прошлом стиле, но с новой логикой
        heatmap_path = self._create_beautiful_dwell_heatmap(
            trajectories, background, width, height, analysis_id, dwell_analysis
        )
        
        # Добавляем информацию о зонах в результаты
        self.dwell_analysis_result = dwell_analysis
        
        zones_count = len(dwell_analysis.get('zones_analysis', []))
        print(f"✅ Тепловая карта создана: {zones_count} активных зон")
        
        # Если нет активных зон, используем fallback
        if zones_count == 0:
            print("⚠️ Нет активных зон, используем простую тепловую карту")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        return heatmap_path
    
    def _create_beautiful_dwell_heatmap(self, trajectories: Dict, background: np.ndarray,
                                       width: int, height: int, analysis_id: str, 
                                       dwell_analysis: Dict) -> str:
        """Создает красивую тепловую карту с анализом времени пребывания в прошлом стиле"""
        
        print("🎨 Создаем красивую тепловую карту с анализом времени пребывания...")
        
        # Проверяем параметры
        try:
            if not isinstance(width, int) or not isinstance(height, int):
                print(f"❌ Некорректные размеры: width={type(width)}, height={type(height)}")
                return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
            
            if not isinstance(background, np.ndarray):
                print(f"❌ Некорректный background: {type(background)}")
                return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
            
            if not isinstance(dwell_analysis, dict):
                print(f"❌ Некорректный dwell_analysis: {type(dwell_analysis)}")
                return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
                
            print(f"✅ Параметры проверены: width={width}, height={height}, background_shape={background.shape}")
        except Exception as e:
            print(f"❌ Ошибка проверки параметров: {e}")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Создаем карту плотности на основе времени пребывания
        density_map = np.zeros((height, width), dtype=np.float32)
        
        # Получаем данные о времени пребывания
        dwell_times = dwell_analysis.get('dwell_times', {})
        grid_info = dwell_analysis.get('grid_info', {})
        zones_analysis = dwell_analysis.get('zones_analysis', [])
        
        if dwell_times and grid_info:
            # Заполняем карту на основе времени пребывания в ячейках
            min_x = grid_info['bounds']['min_x']
            min_y = grid_info['bounds']['min_y']
            cell_size = grid_info['cell_size']
            
            for (cell_x, cell_y), dwell_time in dwell_times.items():
                # Вычисляем координаты ячейки на изображении
                start_x = int(min_x + cell_x * cell_size)
                end_x = int(min_x + (cell_x + 1) * cell_size)
                start_y = int(min_y + cell_y * cell_size)
                end_y = int(min_y + (cell_y + 1) * cell_size)
                
                # Ограничиваем координаты размерами изображения
                start_x = max(0, min(start_x, width - 1))
                end_x = max(0, min(end_x, width))
                start_y = max(0, min(start_y, height - 1))
                end_y = max(0, min(end_y, height))
                
                # Нормализуем время пребывания относительно максимального порога
                max_threshold = self.dwell_time_analyzer.time_thresholds['very_high']
                normalized_intensity = min(dwell_time / max_threshold, 1.0)
                
                # Заполняем ячейку
                if start_y < end_y and start_x < end_x:
                    density_map[start_y:end_y, start_x:end_x] = normalized_intensity
        
        # Если нет данных о времени пребывания, используем стандартный подход
        if density_map.max() == 0:
            print("⚠️ Нет данных о времени пребывания, используем стандартную карту плотности")
            for trajectory in trajectories.values():
                for point in trajectory:
                    x, y = int(point['x']), int(point['y'])
                    if 0 <= x < width and 0 <= y < height:
                        # Добавляем гауссово пятно
                        cv2.circle(density_map, (x, y), 30, 1.0, -1)
        
        # Размытие для плавности
        if density_map.max() > 0:
            density_map = cv2.GaussianBlur(density_map, (61, 61), 0)
            density_map = density_map / density_map.max()
        
        # Создаем matplotlib фигуру в прошлом стиле
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            print(f"✅ Matplotlib фигура создана: {type(fig)}, {type(ax1)}, {type(ax2)}")
        except Exception as e:
            print(f"❌ Ошибка создания matplotlib фигуры: {e}")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Левая панель - оригинальный кадр
        try:
            background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            ax1.imshow(background_rgb)
            ax1.set_title('Оригинальный кадр из видео', fontsize=14, weight='bold')
            ax1.axis('off')
            print("✅ Левая панель настроена")
        except Exception as e:
            print(f"❌ Ошибка настройки левой панели: {e}")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Правая панель - тепловая карта времени пребывания
        try:
            ax2.imshow(background_rgb, alpha=0.7)
            if density_map.max() > 0:
                # Используем красивую цветовую карту
                heatmap = ax2.imshow(density_map, cmap='hot', alpha=0.6, 
                                   extent=[0, width, height, 0])
                plt.colorbar(heatmap, ax=ax2, shrink=0.8, label='Время пребывания (нормализованное)')
                
                # Добавляем информацию о порогах времени
                time_thresholds = self.dwell_time_analyzer.time_thresholds
                ax2.text(10, 30, f"Пороги: 1с={time_thresholds['light']}с, 3с={time_thresholds['medium']}с, 5с={time_thresholds['high']}с, 10с={time_thresholds['very_high']}с", 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                ax2.text(width/2, height/2, 'Нет данных активности', 
                        ha='center', va='center', fontsize=16, weight='bold', color='red')
            
            ax2.set_title('Тепловая карта времени пребывания', fontsize=14, weight='bold')
            ax2.axis('off')
            print("✅ Правая панель настроена")
        except Exception as e:
            print(f"❌ Ошибка настройки правой панели: {e}")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Добавляем информацию о зонах
        try:
            if zones_analysis:
                active_zones_text = f"Активных зон: {len(zones_analysis)}"
                ax2.text(10, height - 30, active_zones_text, 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            plt.suptitle('Анализ времени пребывания людей в зонах', fontsize=16, weight='bold')
            plt.tight_layout()
            print("✅ Заголовок и layout настроены")
        except Exception as e:
            print(f"❌ Ошибка настройки заголовка: {e}")
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
        
        # Сохраняем
        try:
            output_path = f"static/images/heatmap_dwell_time_{analysis_id}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"🎨 Красивая тепловая карта сохранена: {output_path}")
            return f"/static/images/heatmap_dwell_time_{analysis_id}.png"
        except Exception as e:
            print(f"❌ Ошибка сохранения тепловой карты: {e}")
            plt.close()  # Закрываем фигуру в любом случае
            return self._create_simple_heatmap(trajectories, background, width, height, analysis_id)
    
    def _create_simple_heatmap(self, trajectories: Dict, background: np.ndarray,
                              width: int, height: int, analysis_id: str) -> str:
        """Создает простую тепловую карту как fallback"""
        
        # Создаем карту плотности
        density_map = np.zeros((height, width), dtype=np.float32)
        
        for trajectory in trajectories.values():
            for point in trajectory:
                x, y = int(point['x']), int(point['y'])
                if 0 <= x < width and 0 <= y < height:
                    # Добавляем гауссово пятно
                    cv2.circle(density_map, (x, y), 30, 1.0, -1)
        
        # Размытие для плавности
        if density_map.max() > 0:
            density_map = cv2.GaussianBlur(density_map, (61, 61), 0)
            density_map = density_map / density_map.max()
        
        # Создаем matplotlib фигуру
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Левая панель - оригинальный кадр
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        ax1.imshow(background_rgb)
        ax1.set_title('Оригинальный кадр из видео', fontsize=14)
        ax1.axis('off')
        
        # Правая панель - тепловая карта
        ax2.imshow(background_rgb, alpha=0.7)
        if density_map.max() > 0:
            heatmap = ax2.imshow(density_map, cmap='hot', alpha=0.6, 
                               extent=[0, width, height, 0])
            plt.colorbar(heatmap, ax=ax2, shrink=0.8, label='Интенсивность активности')
        
        ax2.set_title('Тепловая карта активности', fontsize=14)
        ax2.axis('off')
        
        plt.suptitle('Анализ зон активности (fallback)', fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Сохраняем
        output_path = f"static/images/heatmap_{analysis_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"/static/images/heatmap_{analysis_id}.png"
    
    def _create_paths_visualization(self, trajectories: Dict, background: np.ndarray,
                                  width: int, height: int, analysis_id: str) -> str:
        """Создает визуализацию троп на фоне реального видео"""
        
        # Копируем фон
        result_frame = background.copy()
        
        # Цвета для траекторий
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
        ]
        
        # Рисуем каждую траекторию
        for i, (person_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            color = colors[i % len(colors)]
            
            # Сглаживаем траекторию
            smoothed_trajectory = self.trajectory_smoother.smooth_trajectory(trajectory)
            
            # Рисуем сглаженный путь
            if len(smoothed_trajectory) >= 2:
                points = [(int(p['x']), int(p['y'])) for p in smoothed_trajectory]
                
                # Рисуем плавные линии
                for j in range(1, len(points)):
                    cv2.line(result_frame, points[j-1], points[j], color, 3)
            else:
                # Fallback на оригинальные точки
                points = [(int(p['x']), int(p['y'])) for p in trajectory]
                for j in range(1, len(points)):
                    cv2.line(result_frame, points[j-1], points[j], color, 3)
            
            # Начальная точка (зеленый круг)
            cv2.circle(result_frame, points[0], 8, (0, 255, 0), -1)
            cv2.circle(result_frame, points[0], 10, (255, 255, 255), 2)
            
            # Конечная точка (красный квадрат)
            end_point = points[-1]
            cv2.rectangle(result_frame, 
                         (end_point[0]-8, end_point[1]-8),
                         (end_point[0]+8, end_point[1]+8),
                         (0, 0, 255), -1)
            cv2.rectangle(result_frame, 
                         (end_point[0]-10, end_point[1]-10),
                         (end_point[0]+10, end_point[1]+10),
                         (255, 255, 255), 2)
        
        # Добавляем легенду
        legend_y = 30
        cv2.rectangle(result_frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(result_frame, (10, 10), (300, 80), (255, 255, 255), 2)
        
        cv2.putText(result_frame, "LEGENDA:", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(result_frame, (30, 50), 8, (0, 255, 0), -1)
        cv2.putText(result_frame, "Start", (50, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(result_frame, (22, 62), (38, 78), (0, 0, 255), -1)
        cv2.putText(result_frame, "End", (50, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Информация о количестве траекторий
        cv2.putText(result_frame, f"Trajectories: {len(trajectories)}", 
                   (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Сохраняем
        output_path = f"static/images/paths_{analysis_id}.png"
        cv2.imwrite(output_path, result_frame)
        
        return f"/static/images/paths_{analysis_id}.png"
    
    def _create_queue_visualization(self, people_per_frame: List, analysis_id: str) -> str:
        """Создает график загруженности"""
        
        if not people_per_frame:
            return None
        
        # people_per_frame теперь список чисел (количество людей в каждом кадре)
        timestamps = list(range(len(people_per_frame)))
        counts = people_per_frame
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, counts, linewidth=2, color='blue', marker='o', markersize=4)
        plt.fill_between(timestamps, counts, alpha=0.3, color='lightblue')
        
        # Статистики
        if counts:
            avg_count = np.mean(counts)
            max_count = max(counts)
            
            plt.axhline(y=avg_count, color='green', linestyle='--', 
                       label=f'Среднее: {avg_count:.1f}')
            plt.axhline(y=max_count, color='red', linestyle='--', 
                       label=f'Максимум: {max_count}')
        
        plt.xlabel('Номер кадра')
        plt.ylabel('Количество людей')
        plt.title('Загруженность по кадрам (реальные данные)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохраняем
        output_path = f"static/images/queue_{analysis_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"/static/images/queue_{analysis_id}.png"
    
    def _generate_analytics(self, trajectories: Dict, people_per_frame: List,
                          duration: float, visualizations: Dict) -> Dict:
        """Генерирует аналитику на основе реальных данных"""
        
        # Базовая статистика
        total_people = len(trajectories)
        people_counts = people_per_frame  # Теперь это список чисел
        
        max_concurrent = max(people_counts) if people_counts else 0
        avg_concurrent = np.mean(people_counts) if people_counts else 0
        
        # Анализ траекторий
        durations = []
        for trajectory in trajectories.values():
            if len(trajectory) > 1:
                traj_duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
                durations.append(traj_duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        # Время пика
        peak_time = "N/A"
        if people_counts:
            peak_frame_idx = people_counts.index(max_concurrent)
            # Предполагаем 30 FPS для расчета времени
            peak_timestamp = peak_frame_idx / 30.0
            peak_time = f"{int(peak_timestamp // 60):02d}:{int(peak_timestamp % 60):02d}"
        
        # Найдем горячие точки
        hot_spots = self._find_hot_spots(trajectories)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "video_filename": os.path.basename(self.current_video_path) if hasattr(self, 'current_video_path') else "unknown",
            "summary": {
                "total_visitors": total_people,
                "max_concurrent_visitors": max_concurrent,
                "avg_concurrent_visitors": round(avg_concurrent, 1),
                "avg_visit_duration": round(avg_duration, 1),
                "video_duration": round(duration, 1),
                "peak_time": peak_time
            },
            "trajectories": trajectories,  # Добавляем траектории для системы оценки
            "heatmap": {
                "image_path": visualizations.get('heatmap'),
                "hot_spots": hot_spots,
                "description": "Тепловая карта основана на реальных данных из видео"
            },
            "desire_paths": {
                "image_path": visualizations.get('paths'),
                "total_paths": total_people,
                "avg_path_duration": round(avg_duration, 1),
                "description": "Реальные тропы движения людей из загруженного видео"
            },
            "queue_analysis": {
                "image_path": visualizations.get('queue'),
                "max_concurrent": max_concurrent,
                "avg_concurrent": round(avg_concurrent, 1),
                "description": "График загруженности основан на подсчете людей в каждом кадре"
            },
            "anomalies": {
                "total_anomalies": 0,
                "anomalies": [],
                "description": "Детекция аномалий будет добавлена в следующей версии"
            }
        }
    
    def _find_hot_spots(self, trajectories: Dict) -> List[Dict]:
        """Находит зоны повышенной активности в реальных данных"""
        if not trajectories:
            return []
        
        # Собираем все точки
        all_points = []
        for trajectory in trajectories.values():
            for point in trajectory:
                all_points.append((point['x'], point['y']))
        
        if not all_points:
            return []
        
        # Простой анализ плотности
        # Делим пространство на сетку и считаем точки в каждой ячейке
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        grid_size = 10
        x_step = (max_x - min_x) / grid_size
        y_step = (max_y - min_y) / grid_size
        
        grid_counts = {}
        
        for x, y in all_points:
            grid_x = int((x - min_x) // x_step) if x_step > 0 else 0
            grid_y = int((y - min_y) // y_step) if y_step > 0 else 0
            key = (grid_x, grid_y)
            grid_counts[key] = grid_counts.get(key, 0) + 1
        
        # Находим топ-3 зоны
        sorted_zones = sorted(grid_counts.items(), key=lambda x: x[1], reverse=True)
        
        hot_spots = []
        for i, ((grid_x, grid_y), count) in enumerate(sorted_zones[:3]):
            intensity = count / len(all_points)
            hot_spots.append({
                "rank": i + 1,
                "intensity": round(intensity, 2),
                "location": f"Зона {i+1} (активность: {count} точек)",
                "count": count
            })
        
        return hot_spots
