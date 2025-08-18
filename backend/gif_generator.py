import cv2
import numpy as np
from typing import List, Dict, Tuple
import os
from PIL import Image
import imageio

class TrajectoryGifGenerator:
    """Генератор GIF анимаций для оценки траекторий"""
    
    def __init__(self, output_dir: str = "static/trajectory_gifs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("🎬 TrajectoryGifGenerator инициализирован")
    
    def create_trajectory_gif(self, video_path: str, trajectory: List[Dict], 
                             trajectory_id: int, smoothness_factor: float = 0.1,
                             duration_per_frame: float = 0.2) -> str:
        """
        Создает GIF с одной траекторией для оценки
        
        Args:
            video_path: Путь к видео файлу
            trajectory: Список точек траектории
            trajectory_id: ID траектории
            smoothness_factor: Коэффициент плавности
            duration_per_frame: Длительность каждого кадра в секундах
            
        Returns:
            Путь к созданному GIF файлу
        """
        try:
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Не удалось открыть видео")
            
            # Получаем параметры видео
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Создаем список кадров для GIF
            frames = []
            
            # Находим диапазон кадров для траектории
            start_frame = min(point['frame'] for point in trajectory)
            end_frame = max(point['frame'] for point in trajectory)
            
            print(f"🎬 Создаем GIF для траектории {trajectory_id}: кадры {start_frame}-{end_frame}")
            
            # Обрабатываем каждый кадр в диапазоне траектории
            for frame_num in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Рисуем траекторию до текущего кадра
                self._draw_trajectory_progress(frame, trajectory, frame_num, smoothness_factor)
                
                # Добавляем информацию о траектории
                self._add_trajectory_info(frame, trajectory_id, frame_num, len(trajectory))
                
                # Конвертируем BGR в RGB для PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                frames.append(pil_frame)
            
            cap.release()
            
            if not frames:
                raise Exception("Не удалось создать кадры для GIF")
            
            # Создаем имя файла
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            gif_filename = f"{video_name}_trajectory_{trajectory_id}.gif"
            gif_path = os.path.join(self.output_dir, gif_filename)
            
            # Сохраняем GIF
            imageio.mimsave(gif_path, frames, duration=duration_per_frame)
            
            print(f"✅ GIF создан: {gif_path} ({len(frames)} кадров)")
            
            # Возвращаем URL для веб-страницы
            web_path = f"/static/trajectory_gifs/{gif_filename}"
            return web_path
            
        except Exception as e:
            print(f"❌ Ошибка создания GIF: {e}")
            return ""
    
    def _draw_trajectory_progress(self, frame: np.ndarray, trajectory: List[Dict], 
                                 current_frame: int, smoothness_factor: float):
        """Рисует прогресс траектории до текущего кадра"""
        if len(trajectory) < 2:
            return
        
        # Фильтруем точки до текущего кадра
        current_points = [p for p in trajectory if p['frame'] <= current_frame]
        
        if len(current_points) < 2:
            return
        
        # Рисуем траекторию
        for i in range(len(current_points) - 1):
            pt1 = (current_points[i]['x'], current_points[i]['y'])
            pt2 = (current_points[i + 1]['x'], current_points[i + 1]['y'])
            
            # Цвет зависит от прогресса
            progress = i / (len(current_points) - 1)
            color = self._get_progress_color(progress)
            
            # Толщина линии зависит от плавности
            thickness = max(2, int(5 * (1 - smoothness_factor)))
            
            cv2.line(frame, pt1, pt2, color, thickness)
            
            # Рисуем точку в текущей позиции
            if i == len(current_points) - 2:  # Последняя точка
                cv2.circle(frame, pt2, 8, (0, 255, 0), -1)  # Зеленая точка
                cv2.circle(frame, pt2, 8, (0, 0, 0), 2)     # Черная обводка
    
    def _get_progress_color(self, progress: float) -> Tuple[int, int, int]:
        """Возвращает цвет в зависимости от прогресса траектории"""
        # От синего (начало) до красного (конец)
        blue = int(255 * (1 - progress))
        red = int(255 * progress)
        green = 0
        
        return (blue, green, red)  # BGR для OpenCV
    
    def _add_trajectory_info(self, frame: np.ndarray, trajectory_id: int, 
                            current_frame: int, total_points: int):
        """Добавляет информацию о траектории на кадр"""
        # Создаем фон для текста
        text_bg = np.zeros((80, 300, 3), dtype=np.uint8)
        text_bg[:] = (0, 0, 0)  # Черный фон
        
        # Добавляем текст
        cv2.putText(text_bg, f"Траектория {trajectory_id}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(text_bg, f"Кадр: {current_frame}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(text_bg, f"Точек: {total_points}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Размещаем в левом верхнем углу
        h, w = text_bg.shape[:2]
        frame[10:10+h, 10:10+w] = text_bg
    
    def create_comparison_gif(self, video_path: str, original_trajectory: List[Dict], 
                             smoothed_trajectory: List[Dict], trajectory_id: int) -> str:
        """
        Создает GIF для сравнения оригинальной и сглаженной траектории
        
        Args:
            video_path: Путь к видео
            original_trajectory: Оригинальная траектория
            smoothed_trajectory: Сглаженная траектория
            trajectory_id: ID траектории
            
        Returns:
            Путь к GIF файлу сравнения
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Не удалось открыть видео")
            
            frames = []
            start_frame = min(point['frame'] for point in original_trajectory)
            end_frame = max(point['frame'] for point in original_trajectory)
            
            print(f"🔄 Создаем GIF сравнения для траектории {trajectory_id}")
            
            for frame_num in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Рисуем оригинальную траекторию (красная, пунктирная)
                self._draw_trajectory_comparison(frame, original_trajectory, frame_num, 
                                               color=(0, 0, 255), is_dashed=True)
                
                # Рисуем сглаженную траекторию (зеленая, сплошная)
                self._draw_trajectory_comparison(frame, smoothed_trajectory, frame_num, 
                                               color=(0, 255, 0), is_dashed=False)
                
                # Добавляем легенду
                self._add_comparison_legend(frame)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
            
            cap.release()
            
            if not frames:
                raise Exception("Не удалось создать кадры для сравнения")
            
            # Сохраняем GIF сравнения
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            gif_filename = f"{video_name}_comparison_{trajectory_id}.gif"
            gif_path = os.path.join(self.output_dir, gif_filename)
            
            imageio.mimsave(gif_path, frames, duration=0.3)
            
            print(f"✅ GIF сравнения создан: {gif_path}")
            return gif_path
            
        except Exception as e:
            print(f"❌ Ошибка создания GIF сравнения: {e}")
            return ""
    
    def _draw_trajectory_comparison(self, frame: np.ndarray, trajectory: List[Dict], 
                                   current_frame: int, color: Tuple[int, int, int], 
                                   is_dashed: bool):
        """Рисует траекторию для сравнения"""
        current_points = [p for p in trajectory if p['frame'] <= current_frame]
        
        if len(current_points) < 2:
            return
        
        for i in range(len(current_points) - 1):
            pt1 = (current_points[i]['x'], current_points[i]['y'])
            pt2 = (current_points[i + 1]['x'], current_points[i + 1]['y'])
            
            if is_dashed and i % 2 == 0:  # Пунктирная линия
                continue
            
            cv2.line(frame, pt1, pt2, color, 3)
            
            # Точка в текущей позиции
            if i == len(current_points) - 2:
                cv2.circle(frame, pt2, 6, color, -1)
    
    def _add_comparison_legend(self, frame: np.ndarray):
        """Добавляет легенду для сравнения траекторий"""
        legend_bg = np.zeros((60, 250, 3), dtype=np.uint8)
        legend_bg[:] = (0, 0, 0)
        
        # Оригинальная траектория (красная, пунктирная)
        cv2.putText(legend_bg, "--- Оригинал", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Сглаженная траектория (зеленая, сплошная)
        cv2.putText(legend_bg, "___ Сглаженная", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Размещаем в правом верхнем углу
        h, w = legend_bg.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        frame[10:10+h, frame_w-w-10:frame_w-10] = legend_bg
