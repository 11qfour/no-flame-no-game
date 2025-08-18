import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Dict, Tuple
import cv2
import math

class TrajectorySmoother:
    """Сглаживание траекторий движения людей"""
    
    def __init__(self, smoothness_factor: float = 0.1):
        """
        Args:
            smoothness_factor: Коэффициент плавности (0.01 - очень плавно, 1.0 - почти прямые)
        """
        self.smoothness_factor = max(0.01, min(1.0, smoothness_factor))
        print(f"🎨 TrajectorySmoother инициализирован с плавностью: {self.smoothness_factor}")
    
    def smooth_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Сглаживает траекторию движения
        
        Args:
            trajectory: Список точек траектории [{'x': x, 'y': y, 'frame': frame}, ...]
            
        Returns:
            Сглаженная траектория с интерполированными точками
        """
        if len(trajectory) < 3:
            return trajectory  # Нечего сглаживать
        
        try:
            # Проверяем и фильтруем валидные точки
            valid_points = []
            for point in trajectory:
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                    try:
                        x = float(point['x'])
                        y = float(point['y'])
                        if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                            valid_points.append(point)
                    except (ValueError, TypeError):
                        continue
            
            if len(valid_points) < 3:
                print(f"⚠️ Недостаточно валидных точек для сглаживания: {len(valid_points)}")
                return trajectory
            
            # Консервативная предварительная очистка
            cleaned_points = self._remove_outliers(valid_points)
            cleaned_points = self._trim_static_segments(cleaned_points)
            cleaned_points = self._simplify_rdp(cleaned_points)
            if len(cleaned_points) < 3:
                return cleaned_points if cleaned_points else trajectory

            # Консервативное сглаживание без интерполяции (без перерисовки назад/вперед)
            smoothed_trajectory = self._moving_average(cleaned_points)
            return smoothed_trajectory
            
        except Exception as e:
            print(f"⚠️ Ошибка сглаживания: {e}, возвращаем исходную траекторию")
            return trajectory
    
    def _simple_smoothing(self, points: List[Dict]) -> List[Dict]:
        """Простое сглаживание как fallback"""
        try:
            if len(points) < 2:
                return points
            
            # Простое сглаживание: добавляем промежуточные точки
            smoothed = []
            for i in range(len(points) - 1):
                current = points[i]
                next_point = points[i + 1]
                
                # Добавляем текущую точку
                smoothed.append(current)
                
                # Добавляем промежуточную точку
                mid_x = (current['x'] + next_point['x']) // 2
                mid_y = (current['y'] + next_point['y']) // 2
                mid_frame = (current.get('frame', 0) + next_point.get('frame', 0)) // 2
                mid_timestamp = (current.get('timestamp', 0) + next_point.get('timestamp', 0)) / 2
                
                smoothed.append({
                    'x': mid_x,
                    'y': mid_y,
                    'frame': mid_frame,
                    'timestamp': mid_timestamp
                })
            
            # Добавляем последнюю точку
            smoothed.append(points[-1])
            
            print(f"🔄 Простое сглаживание: {len(points)} → {len(smoothed)} точек")
            return smoothed
            
        except Exception as e:
            print(f"⚠️ Ошибка простого сглаживания: {e}")
            return points

    def _remove_outliers(self, points: List[Dict]) -> List[Dict]:
        """Удаляет явные выбросы по скорости и резким разворотам"""
        if len(points) < 3:
            return points
        coords = [(float(p['x']), float(p['y'])) for p in points]
        # Скорости между соседними точками
        step_distances = [math.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1]) for i in range(len(coords)-1)]
        median_step = np.median(step_distances) if step_distances else 0
        if median_step == 0:
            median_step = 1.0
        # Более мягкая фильтрация выбросов
        max_allowed = max(10 * median_step, 50.0)
        filtered = [points[0]]
        for i in range(1, len(points)-1):
            prev_pt = coords[i-1]
            curr_pt = coords[i]
            next_pt = coords[i+1]
            d_prev = math.hypot(curr_pt[0]-prev_pt[0], curr_pt[1]-prev_pt[1])
            d_next = math.hypot(next_pt[0]-curr_pt[0], next_pt[1]-curr_pt[1])
            # Угол поворота
            v1 = (curr_pt[0]-prev_pt[0], curr_pt[1]-prev_pt[1])
            v2 = (next_pt[0]-curr_pt[0], next_pt[1]-curr_pt[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            angle = 0
            if n1*n2 > 0:
                cos_a = max(-1.0, min(1.0, dot/(n1*n2)))
                angle = math.degrees(math.acos(cos_a))
            # Фильтруем точку, если слишком далеко/быстро или разворот слишком резкий при малых шагах
            if d_prev > max_allowed or d_next > max_allowed:
                continue
            if angle > 170 and (d_prev < 8 and d_next < 8):
                continue
            filtered.append(points[i])
        filtered.append(points[-1])
        return filtered

    def _simplify_rdp(self, points: List[Dict]) -> List[Dict]:
        """Упрощает ломаную алгоритмом Рамера—Дугласа—Пекера, сохраняя форму"""
        if len(points) < 3:
            return points
        epsilon = max(1.5, (1.0 - self.smoothness_factor) * 8.0)

        def point_line_distance(px, py, ax, ay, bx, by):
            if ax == bx and ay == by:
                return math.hypot(px-ax, py-ay)
            t = ((px-ax)*(bx-ax) + (py-ay)*(by-ay)) / (((bx-ax)**2) + ((by-ay)**2))
            t = max(0.0, min(1.0, t))
            proj_x = ax + t*(bx-ax)
            proj_y = ay + t*(by-ay)
            return math.hypot(px-proj_x, py-proj_y)

        def rdp(indices_start: int, indices_end: int, keep: List[bool]):
            ax, ay = points[indices_start]['x'], points[indices_start]['y']
            bx, by = points[indices_end]['x'], points[indices_end]['y']
            max_dist = -1.0
            index = -1
            for i in range(indices_start+1, indices_end):
                px, py = points[i]['x'], points[i]['y']
                d = point_line_distance(px, py, ax, ay, bx, by)
                if d > max_dist:
                    max_dist = d
                    index = i
            if max_dist > epsilon and index != -1:
                rdp(indices_start, index, keep)
                rdp(index, indices_end, keep)
            else:
                keep[indices_start] = True
                keep[indices_end] = True

        keep_mask = [False] * len(points)
        rdp(0, len(points)-1, keep_mask)
        simplified = [pt for pt, k in zip(points, keep_mask) if k]
        # Гарантируем наличие крайних точек
        if not simplified or simplified[0] != points[0]:
            simplified = [points[0]] + simplified
        if simplified[-1] != points[-1]:
            simplified.append(points[-1])
        return simplified

    def _trim_static_segments(self, points: List[Dict]) -> List[Dict]:
        """Убирает начальные и конечные участки с микродвижениями, чтобы маркеры Start/End были на реальных началах/концах"""
        if len(points) < 3:
            return points
        coords = [(float(p['x']), float(p['y'])) for p in points]
        # Если вся траектория почти статична — ничего не трогаем
        total = sum(math.hypot(coords[i][0]-coords[i-1][0], coords[i][1]-coords[i-1][1]) for i in range(1, len(coords)))
        if total < 40.0:
            return points
        # Порог движения (пиксели)
        threshold = 15.0
        # Находим первый индекс, где накопленное смещение превысило порог
        acc = 0.0
        start_idx = 0
        for i in range(1, len(coords)):
            acc += math.hypot(coords[i][0]-coords[i-1][0], coords[i][1]-coords[i-1][1])
            if acc >= threshold:
                start_idx = i-1
                break
        # С конца
        acc = 0.0
        end_idx = len(coords)-1
        for i in range(len(coords)-1, 0, -1):
            acc += math.hypot(coords[i][0]-coords[i-1][0], coords[i][1]-coords[i-1][1])
            if acc >= threshold:
                end_idx = i
                break
        if start_idx >= end_idx:
            return points
        return points[start_idx:end_idx+1]

    def _moving_average(self, points: List[Dict], window: int = 5) -> List[Dict]:
        """Плавное сглаживание скользящим окном без интерполяции и без изменения длины"""
        n = len(points)
        if n <= 2:
            return points
        w = max(3, min(window, n if n % 2 == 1 else n-1))
        half = w // 2
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        smoothed = []
        for i in range(n):
            left = max(0, i - half)
            right = min(n, i + half + 1)
            avg_x = int(np.mean(xs[left:right]))
            avg_y = int(np.mean(ys[left:right]))
            smoothed.append({
                'x': avg_x,
                'y': avg_y,
                'frame': points[i].get('frame', 0),
                'timestamp': points[i].get('timestamp', 0)
            })
        return smoothed
    
    def adjust_smoothness(self, new_smoothness: float):
        """Изменяет коэффициент плавности"""
        self.smoothness_factor = max(0.01, min(1.0, new_smoothness))
        print(f"🎛️ Плавность изменена на: {self.smoothness_factor}")
    
    def get_smoothness_info(self) -> Dict:
        """Возвращает информацию о текущих настройках плавности"""
        return {
            'smoothness_factor': self.smoothness_factor,
            'description': self._get_smoothness_description()
        }
    
    def _get_smoothness_description(self) -> str:
        """Описание уровня плавности"""
        if self.smoothness_factor < 0.05:
            return "Очень плавные линии"
        elif self.smoothness_factor < 0.2:
            return "Плавные линии"
        elif self.smoothness_factor < 0.5:
            return "Умеренно плавные линии"
        else:
            return "Почти прямые линии"
