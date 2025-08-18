"""
Модуль для обработки видео и детекции людей
Использует OpenCV для трекинга объектов
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import json
import time

class VideoProcessor:
    def __init__(self):
        """Инициализация процессора видео"""
        # Загружаем предтренированную модель для детекции людей
        self.net = cv2.dnn.readNetFromDarknet(
            'static/models/yolo.cfg', 
            'static/models/yolo.weights'
        ) if self._check_models_exist() else None
        
        # Если YOLO недоступна, используем HOG детектор
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Трекеры для отслеживания людей
        self.trackers = []
        self.person_trajectories = {}
        self.person_id_counter = 0
        
    def _check_models_exist(self) -> bool:
        """Проверяем наличие файлов модели YOLO"""
        import os
        return (os.path.exists('static/models/yolo.cfg') and 
                os.path.exists('static/models/yolo.weights'))
    
    def detect_people(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция людей на кадре
        Возвращает список bbox в формате (x, y, w, h)
        """
        if self.net is not None:
            return self._detect_with_yolo(frame)
        else:
            return self._detect_with_hog(frame)
    
    def _detect_with_hog(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детекция с помощью HOG дескриптора"""
        # Изменяем размер для ускорения
        small_frame = cv2.resize(frame, (640, 480))
        
        # Детекция людей
        boxes, weights = self.hog.detectMultiScale(
            small_frame, 
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        
        # Масштабируем координаты обратно
        height_ratio = frame.shape[0] / 480
        width_ratio = frame.shape[1] / 640
        
        scaled_boxes = []
        for (x, y, w, h) in boxes:
            scaled_boxes.append((
                int(x * width_ratio),
                int(y * height_ratio),
                int(w * width_ratio),
                int(h * height_ratio)
            ))
        
        return scaled_boxes
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детекция с помощью YOLO (если доступна)"""
        # Реализация YOLO детекции
        # Пока возвращаем пустой список, будет доработано
        return []
    
    def track_people(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> Dict:
        """
        Отслеживание людей между кадрами
        Возвращает данные о траекториях
        """
        current_centroids = []
        
        # Вычисляем центры детекций
        for (x, y, w, h) in detections:
            center_x = x + w // 2
            center_y = y + h // 2
            current_centroids.append((center_x, center_y))
        
        # Простой алгоритм сопоставления по расстоянию
        matched_trajectories = {}
        
        if hasattr(self, 'previous_centroids'):
            for i, current_centroid in enumerate(current_centroids):
                min_distance = float('inf')
                matched_id = None
                
                for person_id, prev_centroid in self.previous_centroids.items():
                    distance = np.sqrt(
                        (current_centroid[0] - prev_centroid[0])**2 + 
                        (current_centroid[1] - prev_centroid[1])**2
                    )
                    
                    if distance < min_distance and distance < 100:  # Порог сопоставления
                        min_distance = distance
                        matched_id = person_id
                
                if matched_id is not None:
                    matched_trajectories[matched_id] = current_centroid
                else:
                    # Новый человек
                    self.person_id_counter += 1
                    matched_trajectories[self.person_id_counter] = current_centroid
        else:
            # Первый кадр
            for i, centroid in enumerate(current_centroids):
                self.person_id_counter += 1
                matched_trajectories[self.person_id_counter] = centroid
        
        self.previous_centroids = matched_trajectories.copy()
        return matched_trajectories
    
    def process_video(self, video_path: str) -> Dict:
        """
        Основная функция обработки видео
        Возвращает данные о траекториях всех людей
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Не удалось открыть видео: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Обрабатываем видео: {total_frames} кадров, {fps} FPS")
        
        frame_count = 0
        tracking_data = {
            'trajectories': {},
            'frame_data': [],
            'metadata': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': total_frames / fps if fps > 0 else 0
            }
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем каждый N-й кадр для ускорения
            if frame_count % 3 != 0:
                continue
            
            # Детекция людей
            detections = self.detect_people(frame)
            
            # Трекинг
            current_positions = self.track_people(frame, detections)
            
            # Сохраняем данные кадра
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps if fps > 0 else 0,
                'people_count': len(current_positions),
                'positions': current_positions
            }
            tracking_data['frame_data'].append(frame_data)
            
            # Обновляем траектории
            for person_id, position in current_positions.items():
                if person_id not in tracking_data['trajectories']:
                    tracking_data['trajectories'][person_id] = []
                
                tracking_data['trajectories'][person_id].append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'x': position[0],
                    'y': position[1]
                })
            
            # Прогресс
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Прогресс обработки: {progress:.1f}%")
        
        cap.release()
        
        print(f"Обработка завершена. Найдено {len(tracking_data['trajectories'])} уникальных людей")
        
        # Сохраняем результаты
        output_path = video_path.replace('.', '_tracking_data.')
        if output_path.endswith('_tracking_data.'):
            output_path += 'json'
        else:
            output_path = output_path.rsplit('.', 1)[0] + '_tracking_data.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_data, f, ensure_ascii=False, indent=2)
        
        return tracking_data
