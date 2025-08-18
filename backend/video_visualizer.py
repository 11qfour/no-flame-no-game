"""
Модуль для создания наглядных визуализаций поверх видео
Показывает тропы людей и зоны задержек прямо на кадрах
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
from datetime import datetime

class VideoVisualizer:
    def __init__(self):
        """Инициализация визуализатора"""
        self.colors = [
            (255, 0, 0),    # Красный
            (0, 255, 0),    # Зеленый  
            (0, 0, 255),    # Синий
            (255, 255, 0),  # Желтый
            (255, 0, 255),  # Пурпурный
            (0, 255, 255),  # Голубой
            (255, 128, 0),  # Оранжевый
            (128, 0, 255),  # Фиолетовый
        ]
        
    def create_demo_video_with_paths(self, output_path: str = "static/images/demo_video_analysis.mp4") -> str:
        """Создает демо-видео с наложенными тропами и зонами задержек"""
        
        # Параметры видео
        width, height = 640, 480
        fps = 10
        duration_seconds = 20
        total_frames = fps * duration_seconds
        
        # Создаем видео writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Генерируем демо-траектории
        trajectories = self._generate_demo_trajectories(total_frames, width, height)
        
        # Создаем каждый кадр
        for frame_idx in range(total_frames):
            frame = self._create_base_frame(width, height)
            
            # Добавляем элементы интерьера
            frame = self._add_interior_elements(frame, width, height)
            
            # Добавляем траектории до текущего кадра
            frame = self._draw_historical_paths(frame, trajectories, frame_idx)
            
            # Добавляем тепловую карту зон задержек
            frame = self._overlay_heat_zones(frame, trajectories, frame_idx, width, height)
            
            # Добавляем текущие позиции людей
            frame = self._draw_current_people(frame, trajectories, frame_idx)
            
            # Добавляем информационную панель
            frame = self._add_info_panel(frame, frame_idx, total_frames)
            
            # Записываем кадр
            out.write(frame)
        
        out.release()
        print(f"✅ Демо-видео создано: {output_path}")
        return output_path
    
    def _generate_demo_trajectories(self, total_frames: int, width: int, height: int) -> Dict:
        """Генерирует реалистичные траектории для демо"""
        trajectories = {}
        
        # Человек 1: Вход -> Касса (быстро)
        person1 = []
        start_x, start_y = 50, 100
        end_x, end_y = 550, 350
        for i in range(total_frames):
            if i < total_frames * 0.6:  # Появляется в первой половине
                progress = (i / (total_frames * 0.6)) ** 0.8
                x = int(start_x + (end_x - start_x) * progress)
                y = int(start_y + (end_y - start_y) * progress)
                # Добавляем небольшие отклонения
                x += int(np.sin(i * 0.3) * 15)
                y += int(np.cos(i * 0.2) * 10)
                person1.append((max(20, min(width-20, x)), max(20, min(height-20, y))))
        trajectories['person1'] = person1
        
        # Человек 2: Вход -> Витрина -> Долгая задержка -> Касса
        person2 = []
        points = [(60, 120), (200, 200), (250, 220), (500, 340)]
        delays = [0, 30, 80, 20]  # Задержки на каждой точке (в кадрах)
        
        current_frame = 20  # Появляется позже
        for i, ((x, y), delay) in enumerate(zip(points, delays)):
            # Движение к точке
            if i > 0:
                prev_x, prev_y = points[i-1]
                move_frames = 15
                for f in range(move_frames):
                    if current_frame < total_frames:
                        progress = f / move_frames
                        curr_x = int(prev_x + (x - prev_x) * progress)
                        curr_y = int(prev_y + (y - prev_y) * progress)
                        person2.append((curr_x, curr_y))
                        current_frame += 1
            
            # Задержка в точке
            for f in range(delay):
                if current_frame < total_frames:
                    # Небольшие колебания во время ожидания
                    noise_x = int(np.random.normal(0, 3))
                    noise_y = int(np.random.normal(0, 3))
                    person2.append((max(20, min(width-20, x + noise_x)), 
                                  max(20, min(height-20, y + noise_y))))
                    current_frame += 1
        
        trajectories['person2'] = person2
        
        # Человек 3: Заходит, делает круг, уходит
        person3 = []
        center_x, center_y = 300, 250
        radius = 80
        start_frame = 40
        
        for i in range(total_frames):
            if i >= start_frame and i < start_frame + 60:
                angle = (i - start_frame) * 0.15
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                person3.append((max(20, min(width-20, x)), max(20, min(height-20, y))))
        
        trajectories['person3'] = person3
        
        return trajectories
    
    def _create_base_frame(self, width: int, height: int) -> np.ndarray:
        """Создает базовый кадр (фон помещения)"""
        # Создаем кадр с цветом пола
        frame = np.full((height, width, 3), (240, 235, 220), dtype=np.uint8)
        
        # Добавляем текстуру пола
        for i in range(0, height, 40):
            cv2.line(frame, (0, i), (width, i), (220, 215, 200), 1)
        for i in range(0, width, 40):
            cv2.line(frame, (i, 0), (i, height), (220, 215, 200), 1)
        
        return frame
    
    def _add_interior_elements(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Добавляет элементы интерьера (мебель, стены)"""
        # Касса (правый верхний угол)
        cv2.rectangle(frame, (520, 320), (600, 380), (100, 150, 200), -1)
        cv2.putText(frame, "KACCA", (525, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Витрина с едой (центр)
        cv2.rectangle(frame, (180, 180), (280, 240), (150, 200, 100), -1)
        cv2.putText(frame, "VITRINA", (185, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Столики
        table_positions = [(100, 300), (400, 150), (450, 300)]
        for x, y in table_positions:
            cv2.circle(frame, (x, y), 20, (139, 69, 19), -1)
        
        # Вход
        cv2.rectangle(frame, (30, 80), (90, 140), (50, 150, 50), -1)
        cv2.putText(frame, "VHOD", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_historical_paths(self, frame: np.ndarray, trajectories: Dict, current_frame: int) -> np.ndarray:
        """Рисует исторические тропы движения"""
        for person_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                color_idx = hash(person_id) % len(self.colors)
                color = self.colors[color_idx]
                
                # Рисуем путь до текущего момента
                points_to_draw = []
                for i, point in enumerate(trajectory):
                    if i <= current_frame:
                        points_to_draw.append(point)
                
                if len(points_to_draw) > 1:
                    # Рисуем линию пути с затуханием
                    for i in range(1, len(points_to_draw)):
                        alpha = (i / len(points_to_draw)) * 0.8 + 0.2
                        thickness = int(3 * alpha)
                        
                        # Затухающий цвет
                        faded_color = tuple(int(c * alpha) for c in color)
                        
                        cv2.line(frame, points_to_draw[i-1], points_to_draw[i], 
                                faded_color, thickness)
        
        return frame
    
    def _overlay_heat_zones(self, frame: np.ndarray, trajectories: Dict, current_frame: int, 
                           width: int, height: int) -> np.ndarray:
        """Накладывает тепловые зоны задержек"""
        # Создаем карту плотности
        density_map = np.zeros((height, width), dtype=np.float32)
        
        for trajectory in trajectories.values():
            for i, (x, y) in enumerate(trajectory):
                if i <= current_frame:
                    # Увеличиваем плотность в радиусе вокруг точки
                    cv2.circle(density_map, (x, y), 25, 1.0, -1)
        
        # Размытие для плавности
        density_map = cv2.GaussianBlur(density_map, (31, 31), 0)
        
        # Нормализация
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        # Создаем цветовую карту (красный для горячих зон)
        heat_overlay = np.zeros_like(frame)
        heat_mask = density_map > 0.3  # Порог для отображения
        
        heat_overlay[heat_mask] = [0, 0, int(255 * density_map[heat_mask].max())]
        
        # Накладываем с прозрачностью
        alpha = 0.3
        frame = cv2.addWeighted(frame, 1-alpha, heat_overlay, alpha, 0)
        
        return frame
    
    def _draw_current_people(self, frame: np.ndarray, trajectories: Dict, current_frame: int) -> np.ndarray:
        """Рисует текущие позиции людей"""
        for person_id, trajectory in trajectories.items():
            if current_frame < len(trajectory):
                x, y = trajectory[current_frame]
                color_idx = hash(person_id) % len(self.colors)
                color = self.colors[color_idx]
                
                # Рисуем человека как круг
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                
                # Добавляем ID
                cv2.putText(frame, person_id[-1], (x-5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _add_info_panel(self, frame: np.ndarray, current_frame: int, total_frames: int) -> np.ndarray:
        """Добавляет информационную панель"""
        height, width = frame.shape[:2]
        
        # Фон панели
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), (255, 255, 255), 2)
        
        # Время
        time_text = f"Время: {current_frame // 10:02d}:{(current_frame % 10) * 6:02d}"
        cv2.putText(frame, time_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Количество людей
        active_people = sum(1 for traj in trajectories.values() if current_frame < len(traj))
        people_text = f"Посетителей: {active_people}"
        cv2.putText(frame, people_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Прогресс
        progress_text = f"Прогресс: {(current_frame/total_frames)*100:.0f}%"
        cv2.putText(frame, progress_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def create_static_visualization(self, output_path: str = "static/images/demo_paths_overlay.png") -> str:
        """Создает статичную визуализацию с наложенными тропами"""
        width, height = 640, 480
        
        # Создаем базовый кадр
        frame = self._create_base_frame(width, height)
        frame = self._add_interior_elements(frame, width, height)
        
        # Генерируем полные траектории
        trajectories = self._generate_demo_trajectories(200, width, height)
        
        # Рисуем все траектории
        for person_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                color_idx = hash(person_id) % len(self.colors)
                color = self.colors[color_idx]
                
                # Рисуем полный путь
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], color, 3)
                
                # Отмечаем начало и конец
                if trajectory:
                    cv2.circle(frame, trajectory[0], 12, (0, 255, 0), -1)  # Зеленый - вход
                    cv2.circle(frame, trajectory[-1], 12, (0, 0, 255), -1)  # Красный - выход
        
        # Добавляем легенду
        cv2.rectangle(frame, (width-200, 10), (width-10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (width-200, 10), (width-10, 120), (255, 255, 255), 2)
        
        cv2.putText(frame, "LEGENDA:", (width-190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (width-180, 45), 6, (0, 255, 0), -1)
        cv2.putText(frame, "Vhod", (width-165, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(frame, (width-180, 65), 6, (0, 0, 255), -1)
        cv2.putText(frame, "Vyhod", (width-165, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.line(frame, (width-185, 85), (width-165, 85), (255, 255, 0), 3)
        cv2.putText(frame, "Tropy", (width-160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Сохраняем
        cv2.imwrite(output_path, frame)
        print(f"✅ Статичная визуализация создана: {output_path}")
        return output_path

# Глобальная переменная для создания демо-данных
trajectories = {}
