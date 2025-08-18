"""
Модуль для генерации аналитики из данных трекинга
Создает тепловые карты, анализ очередей, детекцию аномалий
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta
import os

class AnalyticsGenerator:
    def __init__(self):
        """Инициализация генератора аналитики"""
        self.heatmap_resolution = (100, 100)  # Разрешение тепловой карты
        
    def generate_analytics(self, tracking_data: Dict) -> Dict:
        """
        Главная функция генерации аналитики
        """
        analytics = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(tracking_data),
            'heatmap': self._generate_heatmap(tracking_data),
            'desire_paths': self._generate_desire_paths(tracking_data),
            'queue_analysis': self._analyze_queues(tracking_data),
            'anomalies': self._detect_anomalies(tracking_data),
            'time_analysis': self._analyze_time_patterns(tracking_data)
        }
        
        return analytics
    
    def _generate_summary(self, tracking_data: Dict) -> Dict:
        """Генерация общей сводки"""
        trajectories = tracking_data.get('trajectories', {})
        frame_data = tracking_data.get('frame_data', [])
        metadata = tracking_data.get('metadata', {})
        
        total_visitors = len(trajectories)
        max_concurrent = max([fd.get('people_count', 0) for fd in frame_data]) if frame_data else 0
        avg_concurrent = np.mean([fd.get('people_count', 0) for fd in frame_data]) if frame_data else 0
        
        # Средняя длительность пребывания
        durations = []
        for person_id, trajectory in trajectories.items():
            if len(trajectory) > 1:
                duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
                durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'total_visitors': total_visitors,
            'max_concurrent_visitors': max_concurrent,
            'avg_concurrent_visitors': round(avg_concurrent, 1),
            'avg_visit_duration': round(avg_duration, 1),
            'video_duration': metadata.get('duration', 0),
            'peak_time': self._find_peak_time(frame_data)
        }
    
    def _find_peak_time(self, frame_data: List[Dict]) -> str:
        """Находит время пика посещаемости"""
        if not frame_data:
            return "Неизвестно"
        
        max_people = 0
        peak_frame = 0
        
        for fd in frame_data:
            if fd.get('people_count', 0) > max_people:
                max_people = fd.get('people_count', 0)
                peak_frame = fd.get('frame_number', 0)
        
        peak_timestamp = peak_frame / 30  # Примерно 30 FPS
        minutes = int(peak_timestamp // 60)
        seconds = int(peak_timestamp % 60)
        
        return f"{minutes:02d}:{seconds:02d}"
    
    def _generate_heatmap(self, tracking_data: Dict) -> Dict:
        """Генерация тепловой карты популярности зон"""
        trajectories = tracking_data.get('trajectories', {})
        
        if not trajectories:
            return {'error': 'Нет данных для генерации тепловой карты'}
        
        # Определяем границы видео
        all_x = []
        all_y = []
        
        for trajectory in trajectories.values():
            for point in trajectory:
                all_x.append(point['x'])
                all_y.append(point['y'])
        
        if not all_x or not all_y:
            return {'error': 'Недостаточно данных для тепловой карты'}
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Создаем сетку для тепловой карты
        grid_x = np.linspace(min_x, max_x, self.heatmap_resolution[0])
        grid_y = np.linspace(min_y, max_y, self.heatmap_resolution[1])
        
        heatmap_data = np.zeros(self.heatmap_resolution)
        
        # Заполняем тепловую карту
        for trajectory in trajectories.values():
            for point in trajectory:
                # Находим ближайшую ячейку сетки
                x_idx = np.digitize(point['x'], grid_x) - 1
                y_idx = np.digitize(point['y'], grid_y) - 1
                
                # Проверяем границы
                x_idx = max(0, min(x_idx, self.heatmap_resolution[0] - 1))
                y_idx = max(0, min(y_idx, self.heatmap_resolution[1] - 1))
                
                heatmap_data[y_idx, x_idx] += 1
        
        # Нормализуем данные
        if heatmap_data.max() > 0:
            heatmap_data = heatmap_data / heatmap_data.max()
        
        # Создаем визуализацию
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='hot', interpolation='bilinear', origin='lower')
        plt.colorbar(label='Относительная популярность')
        plt.title('Тепловая карта популярности зон')
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        
        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = f"static/heatmaps/heatmap_{timestamp}.png"
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Находим самые популярные зоны
        hot_spots = self._find_hot_spots(heatmap_data, grid_x, grid_y)
        
        return {
            'image_path': heatmap_path,
            'hot_spots': hot_spots,
            'data_shape': heatmap_data.shape,
            'max_intensity': float(heatmap_data.max())
        }
    
    def _find_hot_spots(self, heatmap_data: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> List[Dict]:
        """Находит самые популярные зоны"""
        # Находим топ-5 самых горячих точек
        flat_indices = np.argsort(heatmap_data.flatten())[-5:]
        hot_spots = []
        
        for idx in reversed(flat_indices):
            y_idx, x_idx = np.unravel_index(idx, heatmap_data.shape)
            intensity = heatmap_data[y_idx, x_idx]
            
            if intensity > 0.1:  # Минимальный порог
                hot_spots.append({
                    'x': float(grid_x[x_idx]),
                    'y': float(grid_y[y_idx]),
                    'intensity': float(intensity),
                    'rank': len(hot_spots) + 1
                })
        
        return hot_spots
    
    def _generate_desire_paths(self, tracking_data: Dict) -> Dict:
        """Генерация карты 'троп желаний' - популярных маршрутов"""
        trajectories = tracking_data.get('trajectories', {})
        
        if not trajectories:
            return {'error': 'Нет данных для анализа маршрутов'}
        
        # Создаем визуализацию траекторий
        plt.figure(figsize=(12, 8))
        
        path_data = []
        colors_list = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
        
        for i, (person_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            x_coords = [point['x'] for point in trajectory]
            y_coords = [point['y'] for point in trajectory]
            
            # Рисуем траекторию
            plt.plot(x_coords, y_coords, alpha=0.6, linewidth=2, color=colors_list[i])
            
            # Отмечаем начальную и конечную точки
            plt.scatter(x_coords[0], y_coords[0], c='green', s=50, alpha=0.8)  # Вход
            plt.scatter(x_coords[-1], y_coords[-1], c='red', s=50, alpha=0.8)  # Выход
            
            path_data.append({
                'person_id': person_id,
                'start': {'x': x_coords[0], 'y': y_coords[0]},
                'end': {'x': x_coords[-1], 'y': y_coords[-1]},
                'length': len(trajectory),
                'duration': trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
            })
        
        plt.title('Карта "троп желаний" - маршруты посетителей')
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        plt.legend(['Траектории', 'Входы', 'Выходы'], loc='upper right')
        
        # Сохраняем изображение
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths_image = f"static/heatmaps/desire_paths_{timestamp}.png"
        plt.savefig(paths_image, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Анализируем паттерны
        common_patterns = self._analyze_movement_patterns(path_data)
        
        return {
            'image_path': paths_image,
            'total_paths': len(path_data),
            'common_patterns': common_patterns,
            'avg_path_duration': np.mean([p['duration'] for p in path_data]) if path_data else 0
        }
    
    def _analyze_movement_patterns(self, path_data: List[Dict]) -> List[Dict]:
        """Анализ общих паттернов движения"""
        if not path_data:
            return []
        
        # Группируем по начальным и конечным точкам
        entry_points = {}
        exit_points = {}
        
        for path in path_data:
            start_key = f"{int(path['start']['x']//50)*50}_{int(path['start']['y']//50)*50}"
            end_key = f"{int(path['end']['x']//50)*50}_{int(path['end']['y']//50)*50}"
            
            entry_points[start_key] = entry_points.get(start_key, 0) + 1
            exit_points[end_key] = exit_points.get(end_key, 0) + 1
        
        # Топ входных точек
        top_entries = sorted(entry_points.items(), key=lambda x: x[1], reverse=True)[:3]
        top_exits = sorted(exit_points.items(), key=lambda x: x[1], reverse=True)[:3]
        
        patterns = []
        
        for i, (point, count) in enumerate(top_entries):
            patterns.append({
                'type': 'popular_entry',
                'rank': i + 1,
                'location': point,
                'count': count,
                'percentage': round((count / len(path_data)) * 100, 1)
            })
        
        for i, (point, count) in enumerate(top_exits):
            patterns.append({
                'type': 'popular_exit',
                'rank': i + 1,
                'location': point,
                'count': count,
                'percentage': round((count / len(path_data)) * 100, 1)
            })
        
        return patterns
    
    def _analyze_queues(self, tracking_data: Dict) -> Dict:
        """Анализ очередей и времени ожидания"""
        frame_data = tracking_data.get('frame_data', [])
        
        if not frame_data:
            return {'error': 'Нет данных для анализа очередей'}
        
        # Анализ количества людей по времени
        time_series = []
        people_counts = []
        
        for fd in frame_data:
            time_series.append(fd.get('timestamp', 0))
            people_counts.append(fd.get('people_count', 0))
        
        # Создаем график загруженности по времени
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, people_counts, linewidth=2, color='blue')
        plt.fill_between(time_series, people_counts, alpha=0.3, color='lightblue')
        plt.title('Загруженность заведения по времени')
        plt.xlabel('Время (секунды)')
        plt.ylabel('Количество посетителей')
        plt.grid(True, alpha=0.3)
        
        # Добавляем пороговые линии
        avg_people = np.mean(people_counts)
        max_people = max(people_counts)
        
        plt.axhline(y=avg_people, color='green', linestyle='--', label=f'Средняя загруженность: {avg_people:.1f}')
        plt.axhline(y=max_people*0.8, color='orange', linestyle='--', label=f'Высокая загруженность: {max_people*0.8:.1f}')
        plt.axhline(y=max_people*0.9, color='red', linestyle='--', label=f'Критическая загруженность: {max_people*0.9:.1f}')
        
        plt.legend()
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        queue_graph = f"static/heatmaps/queue_analysis_{timestamp}.png"
        plt.savefig(queue_graph, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Анализ пиковых периодов
        peak_periods = self._find_peak_periods(time_series, people_counts)
        
        return {
            'image_path': queue_graph,
            'max_concurrent': max_people,
            'avg_concurrent': round(avg_people, 1),
            'peak_periods': peak_periods,
            'congestion_warnings': self._generate_congestion_warnings(time_series, people_counts)
        }
    
    def _find_peak_periods(self, time_series: List[float], people_counts: List[int]) -> List[Dict]:
        """Находит периоды пиковой загруженности"""
        if not time_series or not people_counts:
            return []
        
        avg_people = np.mean(people_counts)
        peak_threshold = avg_people * 1.5
        
        peak_periods = []
        in_peak = False
        peak_start = 0
        
        for i, (time, count) in enumerate(zip(time_series, people_counts)):
            if count >= peak_threshold and not in_peak:
                in_peak = True
                peak_start = time
            elif count < peak_threshold and in_peak:
                in_peak = False
                peak_periods.append({
                    'start_time': self._format_time(peak_start),
                    'end_time': self._format_time(time),
                    'duration': round(time - peak_start, 1),
                    'max_people': max(people_counts[max(0, i-10):i])
                })
        
        return peak_periods[:5]  # Топ-5 пиковых периодов
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в читаемый вид"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _generate_congestion_warnings(self, time_series: List[float], people_counts: List[int]) -> List[str]:
        """Генерирует предупреждения о перегруженности"""
        warnings = []
        
        if not people_counts:
            return warnings
        
        max_people = max(people_counts)
        avg_people = np.mean(people_counts)
        
        # Проверяем различные условия
        if max_people > avg_people * 2:
            warnings.append(f"⚠️ Пиковая загруженность превышает среднюю в {max_people/avg_people:.1f} раз")
        
        # Ищем периоды длительной высокой загруженности
        high_load_threshold = avg_people * 1.3
        consecutive_high = 0
        max_consecutive = 0
        
        for count in people_counts:
            if count >= high_load_threshold:
                consecutive_high += 1
                max_consecutive = max(max_consecutive, consecutive_high)
            else:
                consecutive_high = 0
        
        if max_consecutive > len(people_counts) * 0.2:  # Более 20% времени
            warnings.append(f"⚠️ Длительные периоды высокой загруженности: {(max_consecutive/len(people_counts)*100):.0f}% времени")
        
        return warnings
    
    def _detect_anomalies(self, tracking_data: Dict) -> Dict:
        """Детекция аномального поведения"""
        trajectories = tracking_data.get('trajectories', {})
        frame_data = tracking_data.get('frame_data', [])
        
        anomalies = []
        
        # Аномалия 1: Люди, которые стоят на одном месте слишком долго
        stationary_anomalies = self._detect_stationary_anomalies(trajectories)
        anomalies.extend(stationary_anomalies)
        
        # Аномалия 2: Неожиданные скопления людей
        crowd_anomalies = self._detect_crowd_anomalies(frame_data)
        anomalies.extend(crowd_anomalies)
        
        # Аномалия 3: Необычные траектории
        trajectory_anomalies = self._detect_trajectory_anomalies(trajectories)
        anomalies.extend(trajectory_anomalies)
        
        return {
            'total_anomalies': len(anomalies),
            'anomalies': anomalies,
            'severity_breakdown': self._categorize_anomalies(anomalies)
        }
    
    def _detect_stationary_anomalies(self, trajectories: Dict) -> List[Dict]:
        """Поиск людей, стоящих на месте слишком долго"""
        anomalies = []
        
        for person_id, trajectory in trajectories.items():
            if len(trajectory) < 10:  # Слишком короткая траектория
                continue
            
            # Вычисляем движение
            positions = [(point['x'], point['y']) for point in trajectory]
            total_movement = 0
            
            for i in range(1, len(positions)):
                movement = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                 (positions[i][1] - positions[i-1][1])**2)
                total_movement += movement
            
            avg_movement = total_movement / len(positions)
            duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
            
            # Аномалия: мало движения при долгом присутствии
            if avg_movement < 20 and duration > 30:  # Стоит практически на месте больше 30 секунд
                anomalies.append({
                    'type': 'stationary_person',
                    'person_id': person_id,
                    'duration': duration,
                    'location': f"({int(positions[0][0])}, {int(positions[0][1])})",
                    'severity': 'medium' if duration < 60 else 'high',
                    'description': f"Посетитель стоял на одном месте {duration:.0f} секунд, возможно блокируя проход"
                })
        
        return anomalies
    
    def _detect_crowd_anomalies(self, frame_data: List[Dict]) -> List[Dict]:
        """Поиск неожиданных скоплений людей"""
        anomalies = []
        
        if not frame_data:
            return anomalies
        
        people_counts = [fd.get('people_count', 0) for fd in frame_data]
        avg_people = np.mean(people_counts)
        std_people = np.std(people_counts)
        
        # Ищем всплески
        threshold = avg_people + 2 * std_people
        
        for i, fd in enumerate(frame_data):
            count = fd.get('people_count', 0)
            if count > threshold and count > 5:  # Минимум 5 человек для аномалии
                timestamp = fd.get('timestamp', 0)
                anomalies.append({
                    'type': 'crowd_surge',
                    'timestamp': self._format_time(timestamp),
                    'people_count': count,
                    'expected_count': round(avg_people),
                    'severity': 'high' if count > threshold * 1.5 else 'medium',
                    'description': f"Неожиданное скопление {count} человек в {self._format_time(timestamp)}"
                })
        
        return anomalies[:3]  # Максимум 3 аномалии этого типа
    
    def _detect_trajectory_anomalies(self, trajectories: Dict) -> List[Dict]:
        """Поиск необычных траекторий движения"""
        anomalies = []
        
        # Анализируем длительность траекторий
        durations = []
        for trajectory in trajectories.values():
            if len(trajectory) > 1:
                duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
                durations.append(duration)
        
        if not durations:
            return anomalies
        
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        # Ищем аномально долгие посещения
        long_threshold = avg_duration + 2 * std_duration
        
        for person_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
            
            if duration > long_threshold and duration > 300:  # Более 5 минут
                anomalies.append({
                    'type': 'long_visit',
                    'person_id': person_id,
                    'duration': duration,
                    'expected_duration': round(avg_duration),
                    'severity': 'low',
                    'description': f"Необычно долгое посещение: {duration/60:.1f} минут (среднее: {avg_duration/60:.1f} минут)"
                })
        
        return anomalies[:2]  # Максимум 2 таких аномалии
    
    def _categorize_anomalies(self, anomalies: List[Dict]) -> Dict:
        """Категоризация аномалий по степени важности"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'low')
            severity_counts[severity] += 1
        
        return severity_counts
    
    def _analyze_time_patterns(self, tracking_data: Dict) -> Dict:
        """Анализ временных паттернов активности"""
        frame_data = tracking_data.get('frame_data', [])
        
        if not frame_data:
            return {'error': 'Нет данных для временного анализа'}
        
        # Разбиваем по интервалам (например, по минутам)
        time_intervals = {}
        
        for fd in frame_data:
            timestamp = fd.get('timestamp', 0)
            interval = int(timestamp // 60) * 60  # Округляем до минут
            
            if interval not in time_intervals:
                time_intervals[interval] = []
            
            time_intervals[interval].append(fd.get('people_count', 0))
        
        # Вычисляем статистики по интервалам
        interval_stats = []
        for interval, counts in time_intervals.items():
            interval_stats.append({
                'time': self._format_time(interval),
                'avg_people': round(np.mean(counts), 1),
                'max_people': max(counts),
                'activity_level': self._categorize_activity(np.mean(counts))
            })
        
        # Сортируем по времени
        interval_stats.sort(key=lambda x: x['time'])
        
        return {
            'intervals': interval_stats,
            'busiest_minute': max(interval_stats, key=lambda x: x['max_people']) if interval_stats else None,
            'quietest_minute': min(interval_stats, key=lambda x: x['avg_people']) if interval_stats else None
        }
    
    def _categorize_activity(self, avg_people: float) -> str:
        """Категоризует уровень активности"""
        if avg_people < 2:
            return 'Низкая'
        elif avg_people < 5:
            return 'Средняя'
        elif avg_people < 8:
            return 'Высокая'
        else:
            return 'Очень высокая'
