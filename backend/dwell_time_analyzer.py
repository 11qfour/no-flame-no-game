"""
Модуль для анализа времени пребывания людей в определенных местах
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from datetime import datetime


class DwellTimeAnalyzer:
    """Анализатор времени пребывания людей в определенных местах"""
    
    def __init__(self, grid_size: int = 50, time_thresholds: Dict[str, float] = None):
        """
        Инициализация анализатора
        
        Args:
            grid_size: Размер сетки для анализа (50x50 пикселей)
            time_thresholds: Пороги времени для разных уровней теплоты
        """
        self.grid_size = grid_size
        self.time_thresholds = time_thresholds or {
            'light': 1.0,      # 1 секунда - легкая теплота
            'medium': 3.0,     # 3 секунды - средняя теплота
            'high': 5.0,       # 5 секунд - высокая теплота
            'very_high': 10.0  # 10 секунд - очень высокая теплота
        }
        
        # Цвета для разных уровней теплоты (BGR формат OpenCV)
        self.heat_colors = {
            'light': (0, 100, 255),      # Оранжевый (легкая теплота)
            'medium': (0, 150, 255),     # Желтый (средняя теплота)
            'high': (0, 200, 255),       # Светло-красный (высокая теплота)
            'very_high': (0, 255, 255),  # Красный (очень высокая теплота)
            'none': (128, 128, 128)      # Серый (минимальная активность)
        }
    
    def analyze_dwell_times(self, trajectories: Dict, video_fps: float = 30.0) -> Dict:
        # Проверяем валидность FPS
        if video_fps <= 0:
            print(f"⚠️ Некорректный FPS в DwellTimeAnalyzer: {video_fps}, используем 30")
            video_fps = 30.0
        """
        Анализирует время пребывания людей в разных местах
        
        Args:
            trajectories: Словарь траекторий людей
            video_fps: Частота кадров видео
            
        Returns:
            Словарь с результатами анализа
        """
        if not trajectories:
            return {'error': 'Нет траекторий для анализа'}
        
        # Определяем границы видео
        all_x, all_y = [], []
        for trajectory in trajectories.values():
            for point in trajectory:
                all_x.append(point['x'])
                all_y.append(point['y'])
        
        if not all_x or not all_y:
            return {'error': 'Недостаточно данных для анализа'}
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Создаем сетку для анализа
        grid_width = int((max_x - min_x) / self.grid_size) + 1
        grid_height = int((max_y - min_y) / self.grid_size) + 1
        
        # Словарь для хранения времени пребывания в каждой ячейке
        dwell_times = defaultdict(float)
        
        # Анализируем каждую траекторию
        for person_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Группируем точки по ячейкам сетки
            cell_groups = self._group_points_by_cells(trajectory, min_x, min_y, self.grid_size)
            
            # Вычисляем время пребывания в каждой ячейке
            for cell_key, points in cell_groups.items():
                if len(points) > 1:
                    # Сортируем точки по времени
                    sorted_points = sorted(points, key=lambda p: p.get('frame', 0))
                    
                    # Вычисляем общее время в ячейке
                    total_time = (sorted_points[-1].get('frame', 0) - sorted_points[0].get('frame', 0)) / video_fps
                    
                    # Добавляем к общему времени в ячейке
                    dwell_times[cell_key] += total_time
        
        # Создаем тепловую карту на основе времени пребывания
        heatmap_data = self._create_dwell_time_heatmap(dwell_times, grid_width, grid_height, min_x, min_y)
        
        # Анализируем зоны по времени пребывания
        zones_analysis = self._analyze_dwell_zones(dwell_times, min_x, min_y, self.grid_size)
        
        return {
            'heatmap_data': heatmap_data,
            'dwell_times': dict(dwell_times),
            'zones_analysis': zones_analysis,
            'grid_info': {
                'width': grid_width,
                'height': grid_height,
                'cell_size': self.grid_size,
                'bounds': {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
            }
        }
    
    def _group_points_by_cells(self, trajectory: List[Dict], min_x: float, min_y: float, cell_size: int) -> Dict:
        """Группирует точки траектории по ячейкам сетки"""
        cell_groups = defaultdict(list)
        
        for point in trajectory:
            x, y = point['x'], point['y']
            cell_x = int((x - min_x) / cell_size)
            cell_y = int((y - min_y) / cell_size)
            cell_key = (cell_x, cell_y)
            cell_groups[cell_key].append(point)
        
        return cell_groups
    
    def _create_dwell_time_heatmap(self, dwell_times: Dict, grid_width: int, grid_height: int, 
                                  min_x: float, min_y: float) -> np.ndarray:
        """Создает тепловую карту на основе времени пребывания"""
        heatmap = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # Заполняем тепловую карту
        for (cell_x, cell_y), dwell_time in dwell_times.items():
            if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                # Нормализуем время пребывания относительно максимального порога
                normalized_time = min(dwell_time / self.time_thresholds['very_high'], 1.0)
                heatmap[cell_y, cell_x] = normalized_time
        
        return heatmap
    
    def _analyze_dwell_zones(self, dwell_times: Dict, min_x: float, min_y: float, cell_size: int) -> List[Dict]:
        """Анализирует зоны по времени пребывания"""
        zones = []
        
        for (cell_x, cell_y), dwell_time in dwell_times.items():
            # Определяем уровень теплоты
            heat_level = self._get_heat_level(dwell_time)
            
            # Пропускаем зоны с минимальной активностью
            if heat_level == 'none':
                continue
            
            # Вычисляем координаты центра ячейки
            center_x = min_x + (cell_x + 0.5) * cell_size
            center_y = min_y + (cell_y + 0.5) * cell_size
            
            zones.append({
                'x': center_x,
                'y': center_y,
                'dwell_time': dwell_time,
                'heat_level': heat_level,
                'color': self.heat_colors[heat_level],
                'description': self._get_heat_description(heat_level, dwell_time)
            })
        
        # Сортируем по времени пребывания (от большего к меньшему)
        zones.sort(key=lambda z: z['dwell_time'], reverse=True)
        
        return zones
    
    def _get_heat_level(self, dwell_time: float) -> str:
        """Определяет уровень теплоты на основе времени пребывания"""
        if dwell_time >= self.time_thresholds['very_high']:
            return 'very_high'
        elif dwell_time >= self.time_thresholds['high']:
            return 'high'
        elif dwell_time >= self.time_thresholds['medium']:
            return 'medium'
        elif dwell_time >= self.time_thresholds['light']:
            return 'light'
        else:
            return 'none'
    
    def _get_heat_description(self, heat_level: str, dwell_time: float) -> str:
        """Возвращает описание уровня теплоты"""
        descriptions = {
            'light': f'Легкая активность ({dwell_time:.1f}с)',
            'medium': f'Средняя активность ({dwell_time:.1f}с)',
            'high': f'Высокая активность ({dwell_time:.1f}с)',
            'very_high': f'Очень высокая активность ({dwell_time:.1f}с)',
            'none': f'Минимальная активность ({dwell_time:.1f}с)'
        }
        return descriptions.get(heat_level, 'Неизвестно')
    
    def create_visualization(self, heatmap_data: np.ndarray, background: np.ndarray,
                           analysis_result: Dict, analysis_id: str) -> str:
        """Создает визуализацию тепловой карты с учетом времени пребывания"""
        
        height, width = background.shape[:2]
        
        # Создаем matplotlib фигуру
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        
        # Левая панель - оригинальный кадр
        background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        ax1.imshow(background_rgb)
        ax1.set_title('Оригинальный кадр', fontsize=14, weight='bold')
        ax1.axis('off')
        
        # Средняя панель - тепловая карта времени пребывания
        if heatmap_data.max() > 0:
            # Нормализуем данные для лучшего отображения
            normalized_heatmap = heatmap_data / heatmap_data.max()
            
            # Создаем цветовую карту с учетом порогов
            heatmap = ax2.imshow(normalized_heatmap, cmap='hot', alpha=0.8, 
                               extent=[0, width, height, 0])
            plt.colorbar(heatmap, ax=ax2, shrink=0.8, label='Время пребывания (нормализованное)')
        else:
            ax2.imshow(background_rgb, alpha=0.3)
            ax2.text(width/2, height/2, 'Нет данных активности', 
                    ha='center', va='center', fontsize=16, weight='bold')
        
        ax2.set_title('Тепловая карта времени пребывания', fontsize=14, weight='bold')
        ax2.axis('off')
        
        # Правая панель - статистика по зонам
        zones_analysis = analysis_result.get('zones_analysis', [])
        if zones_analysis:
            # Группируем зоны по уровням теплоты
            heat_levels = defaultdict(list)
            for zone in zones_analysis:
                heat_levels[zone['heat_level']].append(zone)
            
            # Создаем график
            heat_level_names = ['light', 'medium', 'high', 'very_high']
            heat_level_labels = ['Легкая', 'Средняя', 'Высокая', 'Очень высокая']
            zone_counts = [len(heat_levels.get(level, [])) for level in heat_level_names]
            
            bars = ax3.bar(heat_level_labels, zone_counts, 
                          color=['orange', 'yellow', 'red', 'darkred'])
            ax3.set_title('Распределение зон по активности', fontsize=14, weight='bold')
            ax3.set_ylabel('Количество зон')
            ax3.set_xlabel('Уровень активности')
            
            # Добавляем значения на столбцы
            for bar, count in zip(bars, zone_counts):
                if count > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Нет данных о зонах', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=16)
            ax3.set_title('Статистика зон', fontsize=14, weight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Анализ времени пребывания людей в зонах', fontsize=18, weight='bold')
        plt.tight_layout()
        
        # Сохраняем изображение
        output_path = f"static/images/heatmap_dwell_time_{analysis_id}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"/static/images/heatmap_dwell_time_{analysis_id}.png"
    
    def get_heat_level_color(self, dwell_time: float) -> Tuple[int, int, int]:
        """Возвращает цвет для определенного времени пребывания (BGR формат)"""
        heat_level = self._get_heat_level(dwell_time)
        return self.heat_colors.get(heat_level, (0, 0, 0))
