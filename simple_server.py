"""
Упрощенная версия сервера для тестирования
"""

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os
import json
from datetime import datetime
from backend.trajectory_smoother import TrajectorySmoother
from backend.trajectory_evaluator import TrajectoryEvaluator
from backend.gif_generator import TrajectoryGifGenerator

# Создаем экземпляр FastAPI
app = FastAPI(title="Aura - AI Analytics", version="1.0.0")

# Импортируем трекер прогресса
from backend.progress_tracker import progress_tracker

# Проверяем и подключаем статические файлы
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Статические файлы подключены")
    
    # Проверяем наличие ключевых файлов
    static_files = [
        "static/js/script.js",
        "static/css/style.css",
        "static/js/analytics.js",
        "static/js/trajectory_rating.js"
    ]
    
    for file_path in static_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"📁 {file_path}: {file_size} байт")
        else:
            print(f"❌ {file_path}: НЕ НАЙДЕН")
else:
    print("❌ Папка static не найдена")

# Добавляем отладочный middleware для проверки запросов
@app.middleware("http")
async def debug_requests(request: Request, call_next):
    print(f"🔍 Запрос: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"📤 Ответ: {response.status_code}")
    
    # Отключаем кэш для статических файлов
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response

# Подключаем шаблоны
if os.path.exists("templates"):
    templates = Jinja2Templates(directory="templates")
    print("✅ Шаблоны подключены")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get-progress")
async def get_progress():
    """Получить текущий прогресс обработки"""
    progress_data = progress_tracker.get_progress()
    return progress_data

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """Реальная обработка загруженного видео"""
    try:
        # Сохраняем загруженный файл
        upload_path = f"uploads/{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📁 Видео сохранено: {upload_path}")
        
        # Импортируем модули для обработки
        from backend.video_processor import VideoProcessor
        from backend.analytics_generator import AnalyticsGenerator
        from backend.real_video_analyzer import RealVideoAnalyzer
        
        # Создаем анализатор реального видео
        analyzer = RealVideoAnalyzer()
        
        # Обрабатываем реальное видео
        print("🎬 Начинаем анализ видео...")
        analysis_result = analyzer.analyze_video(upload_path)
        
        # Сохраняем результаты анализа для системы оценки
        analysis_data = {
            'video_filename': file.filename,
            'analysis_result': analysis_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохраняем в JSON файл для последующего использования
        analysis_file = f"uploads/analysis_{file.filename}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Результаты анализа сохранены: {analysis_file}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Видео {file.filename} успешно проанализировано",
            "analytics": analysis_result,
            "analysis_file": analysis_file
        })
        
    except Exception as e:
        print(f"❌ Ошибка обработки видео: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка обработки: {str(e)}"
        }, status_code=500)

@app.get("/analytics/{video_id}")
async def get_analytics(video_id: str, request: Request):
    """Страница с результатами аналитики"""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "video_id": video_id
    })

@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    """Демонстрационная страница с наглядными визуализациями"""
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Тестовая страница для отладки"""
    with open("test_server_debug.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/test-progress", response_class=HTMLResponse)
async def test_progress_page(request: Request):
    """Тестовая страница для проверки прогресс-бара"""
    with open("test_progress_fix.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/debug-static")
async def debug_static():
    """Отладочная информация о статических файлах"""
    static_info = {}
    
    static_files = [
        "static/js/script.js",
        "static/css/style.css",
        "static/js/analytics.js",
        "static/js/trajectory_rating.js"
    ]
    
    for file_path in static_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            static_info[file_path] = {
                "exists": True,
                "size": file_size,
                "readable": os.access(file_path, os.R_OK)
            }
        else:
            static_info[file_path] = {
                "exists": False,
                "size": 0,
                "readable": False
            }
    
    return JSONResponse({
        "static_files": static_info,
        "static_dir_exists": os.path.exists("static"),
        "current_working_dir": os.getcwd()
    })

# ===== API для системы оценки траекторий =====

@app.get("/trajectory-rating/{video_filename}/{trajectory_id}")
async def trajectory_rating_page(request: Request, video_filename: str, trajectory_id: int):
    """Страница для оценки конкретной траектории"""
    try:
        # Получаем информацию о траектории
        evaluator = TrajectoryEvaluator()
        gif_generator = TrajectoryGifGenerator()
        
        # Загружаем результаты анализа
        analysis_file = f"uploads/analysis_{video_filename}.json"
        if not os.path.exists(analysis_file):
            return JSONResponse({
                "status": "error",
                "message": "Анализ видео не найден. Сначала загрузите и проанализируйте видео."
            }, status_code=404)
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Отладочная информация
        print(f"🔍 Анализ данных: {analysis_data.keys()}")
        if 'analysis_result' in analysis_data:
            print(f"📊 Результат анализа: {analysis_data['analysis_result'].keys()}")
        
        # Получаем реальные траектории
        trajectories = analysis_data.get('analysis_result', {}).get('trajectories', {})
        if not trajectories:
            # Попробуем альтернативный путь
            trajectories = analysis_data.get('analysis_result', {}).get('desire_paths', {}).get('trajectories', {})
            if not trajectories:
                return JSONResponse({
                    "status": "error",
                    "message": "Траектории не найдены в анализе. Возможно, видео не содержит движущихся людей."
                }, status_code=404)
        
        # Получаем конкретную траекторию
        # Траектории имеют ключи типа "person_1", "person_2", поэтому нужно найти по индексу
        trajectory_keys = list(trajectories.keys())
        if trajectory_id >= len(trajectory_keys):
            return JSONResponse({
                "status": "error",
                "message": f"Траектория {trajectory_id} не найдена. Всего траекторий: {len(trajectory_keys)}"
            }, status_code=404)
        
        trajectory_key = trajectory_keys[trajectory_id]
        trajectory = trajectories[trajectory_key]
        
        print(f"🎯 Найдена траектория: {trajectory_key} (ID: {trajectory_id})")
        
        # Создаем GIF для траектории
        gif_path = gif_generator.create_trajectory_gif(
            f"uploads/{video_filename}", 
            trajectory, 
            trajectory_id
        )
        
        return templates.TemplateResponse("trajectory_rating.html", {
            "request": request,
            "video_name": video_filename,
            "trajectory_id": trajectory_id,
            "total_trajectories": len(trajectories),
            "gif_path": gif_path
        })
        
    except Exception as e:
        print(f"❌ Ошибка загрузки страницы оценки: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }, status_code=500)

@app.post("/api/rate-trajectory")
async def rate_trajectory(request: Request):
    """API для оценки траектории"""
    try:
        data = await request.json()
        print(f"📥 Получены данные для оценки: {data}")
        
        video_filename = data.get('video_filename')
        trajectory_id = data.get('trajectory_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        smoothness_factor = data.get('smoothness_factor', 0.1)
        
        print(f"🔍 Извлеченные данные: video_filename={video_filename}, trajectory_id={trajectory_id}, rating={rating}")
        
        if not video_filename or trajectory_id is None or not rating:
            print(f"❌ Недостаточно данных: video_filename={bool(video_filename)}, trajectory_id={trajectory_id} (тип: {type(trajectory_id)}), rating={bool(rating)}")
            return JSONResponse({
                "status": "error",
                "message": "Недостаточно данных"
            }, status_code=400)
        
        evaluator = TrajectoryEvaluator()
        
        success = evaluator.rate_trajectory(
            video_filename=video_filename,
            trajectory_id=trajectory_id,
            rating=rating,
            comment=comment,
            smoothness_factor=smoothness_factor
        )
        
        if success:
            return JSONResponse({
                "status": "success",
                "message": "Оценка сохранена"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Ошибка сохранения оценки"
            }, status_code=400)
            
    except Exception as e:
        print(f"❌ Ошибка оценки траектории: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }, status_code=500)

@app.post("/api/regenerate-gif")
async def regenerate_gif(request: Request):
    """API для пересоздания GIF с новыми параметрами плавности"""
    try:
        data = await request.json()
        video_filename = data.get('video_filename')
        trajectory_id = data.get('trajectory_id')
        smoothness_factor = data.get('smoothness_factor', 0.1)
        
        if not all([video_filename, trajectory_id]):
            return JSONResponse({
                "status": "error",
                "message": "Недостаточно данных"
            }, status_code=400)
        
        gif_generator = TrajectoryGifGenerator()
        
        # Загружаем реальную траекторию из анализа
        analysis_file = f"uploads/analysis_{video_filename}.json"
        if not os.path.exists(analysis_file):
            return JSONResponse({
                "status": "error",
                "message": "Анализ видео не найден"
            }, status_code=404)
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        trajectories = analysis_data.get('analysis_result', {}).get('trajectories', {})
        if not trajectories:
            # Попробуем альтернативный путь
            trajectories = analysis_data.get('analysis_result', {}).get('desire_paths', {}).get('trajectories', {})
            if not trajectories:
                return JSONResponse({
                    "status": "error",
                    "message": "Траектории не найдены в анализе"
                }, status_code=404)
        
        # Траектории имеют ключи типа "person_1", "person_2", поэтому нужно найти по индексу
        trajectory_keys = list(trajectories.keys())
        if trajectory_id >= len(trajectory_keys):
            return JSONResponse({
                "status": "error",
                "message": f"Траектория {trajectory_id} не найдена. Всего траекторий: {len(trajectory_keys)}"
            }, status_code=404)
        
        trajectory_key = trajectory_keys[trajectory_id]
        trajectory = trajectories[trajectory_key]
        
        print(f"🎯 Найдена траектория для GIF: {trajectory_key} (ID: {trajectory_id})")
        
        gif_path = gif_generator.create_trajectory_gif(
            f"uploads/{video_filename}",
            trajectory,
            trajectory_id,
            smoothness_factor
        )
        
        return JSONResponse({
            "status": "success",
            "gif_path": gif_path
        })
        
    except Exception as e:
        print(f"❌ Ошибка создания GIF: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }, status_code=500)

@app.get("/api/video-statistics/{video_filename}")
async def get_video_statistics(video_filename: str):
    """API для получения статистики оценок видео"""
    try:
        evaluator = TrajectoryEvaluator()
        stats = evaluator.get_video_statistics(video_filename)
        
        return JSONResponse({
            "status": "success",
            "statistics": stats
        })
        
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }, status_code=500)

@app.get("/api/learning-recommendations")
async def get_learning_recommendations():
    """API для получения рекомендаций по обучению"""
    try:
        evaluator = TrajectoryEvaluator()
        recommendations = evaluator.get_learning_recommendations()
        
        return JSONResponse({
            "status": "success",
            "recommendations": recommendations
        })
        
    except Exception as e:
        print(f"❌ Ошибка получения рекомендаций: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка: {str(e)}"
        }, status_code=500)

@app.post("/analyze-demo-video")
async def analyze_demo_video(request: Request):
    """API для анализа демо-видео"""
    try:
        # Получаем данные из запроса
        data = await request.json()
        video_path = data.get('video_path')
        video_name = data.get('video_name')
        
        if not video_path or not os.path.exists(video_path):
            return JSONResponse({
                "status": "error",
                "message": "Демо-видео не найдено"
            }, status_code=404)
        
        print(f"🎬 Анализируем демо-видео: {video_name} ({video_path})")
        
        # Импортируем модули для обработки
        from backend.real_video_analyzer import RealVideoAnalyzer
        
        # Создаем анализатор реального видео
        analyzer = RealVideoAnalyzer()
        
        # Обрабатываем демо-видео
        print("🎬 Начинаем анализ демо-видео...")
        analysis_result = analyzer.analyze_video(video_path)
        
        # Сохраняем результаты анализа для системы оценки
        analysis_data = {
            'video_filename': os.path.basename(video_path),
            'analysis_result': analysis_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохраняем в JSON файл для последующего использования
        analysis_file = f"uploads/analysis_{os.path.basename(video_path)}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Результаты анализа демо-видео сохранены: {analysis_file}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Демо-видео '{video_name}' успешно проанализировано",
            "analysis_result": analysis_result
        })
        
    except Exception as e:
        print(f"❌ Ошибка анализа демо-видео: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Ошибка анализа демо-видео: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    print("🚀 Запускаем Aura сервер...")
    
    # Создаем необходимые папки
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/heatmaps", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("static/trajectory_gifs", exist_ok=True)
    
    print("📂 Структура проекта готова")
    print("🌐 Сервер будет доступен на: http://127.0.0.1:8000")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)