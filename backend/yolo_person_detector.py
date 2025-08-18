"""
Детектор людей на основе YOLO11
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import os

class YOLOPersonDetector:
    def __init__(self, model_path: str = None):
        """
        Инициализация детектора людей YOLO
        
        Args:
            model_path: Путь к модели YOLO (если None, используется стандартная)
        """
        try:
            if model_path and os.path.exists(model_path):
                print(f"🎯 Загружаем кастомную модель YOLO: {model_path}")
                self.model = YOLO(model_path)
            else:
                print("🎯 Загружаем стандартную модель YOLO")
                # Используем стандартную модель YOLO для детекции людей
                # YOLO автоматически скачает модель при первом использовании
                self.model = YOLO('yolov8n.pt')  # Используем доступную модель
            
            # Устанавливаем порог уверенности
            self.confidence_threshold = 0.25  # Уменьшили с 0.3 для лучшего покрытия
            print("✅ YOLO детектор инициализирован")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации YOLO: {e}")
            self.model = None
    
    def detect_people_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция людей в кадре
        
        Args:
            frame: Кадр видео (BGR)
            
        Returns:
            Список детекций людей с координатами и уверенностью
        """
        if self.model is None:
            print("⚠️ YOLO модель не инициализирована")
            return []
        
        try:
            # Конвертируем BGR в RGB для YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Запускаем детекцию
            results = self.model(frame_rgb, verbose=False)
            
            people_detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Получаем координаты и класс
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Проверяем, что это человек (класс 0 в COCO)
                        if class_id == 0 and confidence > self.confidence_threshold:
                            # Конвертируем в формат для нашего трекера
                            detection = {
                                'bbox': {
                                    'x': int(x1),
                                    'y': int(y1),
                                    'width': int(x2 - x1),
                                    'height': int(y2 - y1)
                                },
                                'confidence': confidence,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2)
                            }
                            
                            # Валидация размера детекции
                            if self._validate_detection(detection):
                                people_detections.append(detection)
                                print(f"👤 YOLO детекция: {detection['bbox']['width']}x{detection['bbox']['height']}, уверенность: {confidence:.3f}")
            
            print(f"🚶 YOLO нашел {len(people_detections)} людей")
            return people_detections
            
        except Exception as e:
            print(f"❌ Ошибка YOLO детекции: {e}")
            return []
    
    def _validate_detection(self, detection: Dict) -> bool:
        """
        Валидация детекции по размеру и качеству
        
        Args:
            detection: Детекция для валидации
            
        Returns:
            True если детекция валидна
        """
        bbox = detection['bbox']
        
        # Минимальный размер (слишком маленькие детекции могут быть шумом)
        if bbox['width'] < 20 or bbox['height'] < 40:  # Уменьшили минимальные размеры
            return False
        
        # Максимальный размер (слишком большие могут быть ошибкой)
        if bbox['width'] > 600 or bbox['height'] > 900:  # Увеличили максимальные размеры
            return False
        
        # Проверка уверенности
        if detection['confidence'] < self.confidence_threshold:
            return False
        
        return True
    
    def detect_multiple_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция нескольких людей в кадре (адаптер для совместимости)
        
        Args:
            frame: Кадр видео
            
        Returns:
            Список детекций в формате (x, y, width, height)
        """
        detections = self.detect_people_in_frame(frame)
        
        # Конвертируем в формат для трекера
        formatted_detections = []
        for det in detections:
            bbox = det['bbox']
            formatted_detections.append((
                bbox['x'], bbox['y'], bbox['width'], bbox['height']
            ))
        
        return formatted_detections
    
    def get_detection_info(self) -> str:
        """Возвращает информацию о детекторе"""
        if self.model is None:
            return "YOLO детектор не инициализирован"
        
        model_name = self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else "Стандартная модель"
        return f"YOLO детектор: {model_name}"
    
    def load_custom_model(self, model_path: str) -> bool:
        """
        Загружает кастомную модель YOLO
        
        Args:
            model_path: Путь к файлу модели (.pt)
            
        Returns:
            True если модель загружена успешно
        """
        try:
            if not os.path.exists(model_path):
                print(f"❌ Файл модели не найден: {model_path}")
                return False
            
            print(f"🔄 Загружаем кастомную модель: {model_path}")
            self.model = YOLO(model_path)
            print("✅ Кастомная модель загружена")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки кастомной модели: {e}")
            return False
    
    def set_confidence_threshold(self, threshold: float):
        """
        Устанавливает порог уверенности для детекции
        
        Args:
            threshold: Порог уверенности (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"🎯 Порог уверенности установлен: {threshold}")
        else:
            print("❌ Некорректный порог уверенности (должен быть 0.0 - 1.0)")
    
    def cleanup(self):
        """Освобождает ресурсы"""
        if hasattr(self, 'model') and self.model:
            try:
                # YOLO автоматически освобождает ресурсы
                print("🧹 Ресурсы YOLO освобождены")
            except Exception as e:
                print(f"⚠️ Ошибка освобождения ресурсов: {e}")
