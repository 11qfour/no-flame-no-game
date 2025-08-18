// JavaScript для страницы оценки траекторий

class TrajectoryRating {
    constructor() {
        this.currentRating = 0;
        this.currentTrajectoryId = null;
        this.videoFilename = null;
        this.totalTrajectories = 0;
        this.ratedTrajectories = new Set();
        this.smoothnessFactor = 0.1;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadStatistics();
        this.updateProgress();
    }
    
    bindEvents() {
        // Звезды
        document.querySelectorAll('.stars i').forEach(star => {
            star.addEventListener('click', (e) => this.selectRating(e.target.dataset.rating));
            star.addEventListener('mouseenter', (e) => this.hoverRating(e.target.dataset.rating));
            star.addEventListener('mouseleave', () => this.clearHover());
        });
        
        // Слайдер плавности
        const smoothnessSlider = document.getElementById('smoothness-slider');
        if (smoothnessSlider) {
            smoothnessSlider.addEventListener('input', (e) => this.updateSmoothness(e.target.value));
        }
        
        // Кнопки
        document.getElementById('submit-rating')?.addEventListener('click', () => this.submitRating());
        document.getElementById('regenerate-gif')?.addEventListener('click', () => this.regenerateGif());
        document.getElementById('skip-trajectory')?.addEventListener('click', () => this.skipTrajectory());
        document.getElementById('prev-trajectory')?.addEventListener('click', () => this.navigateTrajectory(-1));
        document.getElementById('next-trajectory')?.addEventListener('click', () => this.navigateTrajectory(1));
        
        // Модальное окно
        document.getElementById('export-results')?.addEventListener('click', () => this.exportResults());
        document.getElementById('view-all-trajectories')?.addEventListener('click', () => this.viewAllTrajectories());
        document.getElementById('upload-new-video')?.addEventListener('click', () => this.uploadNewVideo());
    }
    
    selectRating(rating) {
        this.currentRating = parseInt(rating);
        console.log('⭐ Выбрана оценка:', this.currentRating);
        
        // Обновляем визуал звезд
        document.querySelectorAll('.stars i').forEach((star, index) => {
            if (index < rating) {
                star.classList.add('selected');
                star.classList.remove('active');
            } else {
                star.classList.remove('selected', 'active');
            }
        });
        
        // Обновляем текст
        const ratingText = document.querySelector('.rating-text');
        if (ratingText) {
            const descriptions = {
                1: 'Очень плохо - серьезные проблемы',
                2: 'Плохо - много недостатков',
                3: 'Удовлетворительно - есть проблемы',
                4: 'Хорошо - незначительные недостатки',
                5: 'Отлично - качественная траектория'
            };
            ratingText.textContent = descriptions[rating];
        }
        
        // Активируем кнопку отправки
        const submitBtn = document.getElementById('submit-rating');
        if (submitBtn) {
            submitBtn.disabled = false;
            console.log('✅ Кнопка отправки активирована');
        }
    }
    
    hoverRating(rating) {
        document.querySelectorAll('.stars i').forEach((star, index) => {
            if (index < rating) {
                star.classList.add('active');
            } else {
                star.classList.remove('active');
            }
        });
    }
    
    clearHover() {
        document.querySelectorAll('.stars i').forEach(star => {
            star.classList.remove('active');
        });
    }
    
    updateSmoothness(value) {
        this.smoothnessFactor = parseFloat(value);
        const smoothnessValue = document.getElementById('smoothness-value');
        if (smoothnessValue) {
            smoothnessValue.textContent = value;
        }
    }
    
    async submitRating() {
        console.log('🚀 Попытка отправки оценки:', {
            currentRating: this.currentRating,
            currentTrajectoryId: this.currentTrajectoryId,
            videoFilename: this.videoFilename
        });
        
        console.log('🔍 Проверка данных:', {
            hasRating: !!this.currentRating,
            ratingValue: this.currentRating,
            hasTrajectoryId: this.currentTrajectoryId !== null && this.currentTrajectoryId !== undefined,
            trajectoryIdValue: this.currentTrajectoryId,
            hasVideoFilename: !!this.videoFilename
        });
        
        if (!this.currentRating || this.currentTrajectoryId === null || this.currentTrajectoryId === undefined) {
            console.error('❌ Недостаточно данных для отправки');
            alert('Пожалуйста, выберите оценку');
            return;
        }
        
        const comment = document.getElementById('comment')?.value || '';
        
        try {
            const response = await fetch('/api/rate-trajectory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_filename: this.videoFilename,
                    trajectory_id: this.currentTrajectoryId,
                    rating: this.currentRating,
                    comment: comment,
                    smoothness_factor: this.smoothnessFactor
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Добавляем в список оцененных
                this.ratedTrajectories.add(this.currentTrajectoryId);
                
                // Показываем уведомление
                this.showNotification('✅ Оценка сохранена!', 'success');
                
                // Обновляем статистику
                this.loadStatistics();
                this.updateProgress();
                
                // Переходим к следующей траектории
                setTimeout(() => this.navigateTrajectory(1), 1000);
                
            } else {
                throw new Error('Ошибка сохранения оценки');
            }
            
        } catch (error) {
            console.error('Ошибка:', error);
            this.showNotification('❌ Ошибка сохранения оценки', 'error');
        }
    }
    
    async regenerateGif() {
        if (!this.currentTrajectoryId) return;
        
        try {
            this.showNotification('🔄 Создаем новый GIF...', 'info');
            
            const response = await fetch('/api/regenerate-gif', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_filename: this.videoFilename,
                    trajectory_id: this.currentTrajectoryId,
                    smoothness_factor: this.smoothnessFactor
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Обновляем GIF
                const gifImg = document.getElementById('trajectory-gif');
                if (gifImg && result.gif_path) {
                    gifImg.src = result.gif_path + '?t=' + Date.now(); // Кэш-бастер
                }
                
                this.showNotification('✅ GIF обновлен!', 'success');
                
            } else {
                throw new Error('Ошибка создания GIF');
            }
            
        } catch (error) {
            console.error('Ошибка:', error);
            this.showNotification('❌ Ошибка создания GIF', 'error');
        }
    }
    
    skipTrajectory() {
        this.showNotification('⏭️ Траектория пропущена', 'info');
        setTimeout(() => this.navigateTrajectory(1), 1000);
    }
    
    navigateTrajectory(direction) {
        if (!this.videoFilename) return;
        
        const newTrajectoryId = this.currentTrajectoryId + direction;
        
        // Проверяем границы
        if (newTrajectoryId < 0) {
            this.showNotification('🏁 Это первая траектория', 'info');
            return;
        }
        
        if (newTrajectoryId >= this.totalTrajectories) {
            this.showNotification('🏁 Это последняя траектория', 'info');
            return;
        }
        
        // Переходим к новой траектории
        window.location.href = `/trajectory-rating/${encodeURIComponent(this.videoFilename)}/${newTrajectoryId}`;
    }
    
    async loadStatistics() {
        if (!this.videoFilename) return;
        
        try {
            const response = await fetch(`/api/video-statistics/${encodeURIComponent(this.videoFilename)}`);
            if (response.ok) {
                const stats = await response.json();
                this.updateStatisticsDisplay(stats);
            }
        } catch (error) {
            console.error('Ошибка загрузки статистики:', error);
        }
    }
    
    updateStatisticsDisplay(stats) {
        const totalRated = document.getElementById('total-rated');
        const avgRating = document.getElementById('avg-rating');
        const remaining = document.getElementById('remaining');
        
        if (totalRated) totalRated.textContent = stats.total_rated || 0;
        if (avgRating) avgRating.textContent = stats.average_rating || '0.0';
        if (remaining) remaining.textContent = (this.totalTrajectories - (stats.total_rated || 0));
    }
    
    updateProgress() {
        const progressFill = document.getElementById('progress-fill');
        if (progressFill) {
            const progress = (this.ratedTrajectories.size / this.totalTrajectories) * 100;
            progressFill.style.width = `${progress}%`;
            if (progress >= 100) {
                this.showCompletionModal();
            }
        }
    }
    
    showCompletionModal() {
        const modal = document.getElementById('completion-modal');
        if (modal) {
            modal.style.display = 'block';
            this.populateFinalStats();
            this.showNotification('🎉 Поздравляем! Все траектории оценены!', 'success');
            
            // Добавляем анимацию появления
            modal.style.animation = 'fadeIn 0.5s ease';
        }
    }
    
    populateFinalStats() {
        const finalStats = document.getElementById('final-stats');
        if (finalStats) {
            const avgRating = this.calculateAverageRating();
            const qualityLevel = this.getQualityLevel(avgRating);
            
            finalStats.innerHTML = `
                <li>📹 Видео: <strong>${this.videoFilename}</strong></li>
                <li>🎯 Всего траекторий: <strong>${this.totalTrajectories}</strong></li>
                <li>⭐ Оценено: <strong>${this.ratedTrajectories.size}</strong></li>
                <li>📊 Средняя оценка: <strong>${avgRating.toFixed(1)}/5.0</strong></li>
                <li>🏆 Уровень качества: <strong>${qualityLevel}</strong></li>
                <li>📅 Дата завершения: <strong>${new Date().toLocaleDateString('ru-RU')}</strong></li>
            `;
        }
    }
    
    getQualityLevel(averageRating) {
        if (averageRating >= 4.5) return 'Отличное качество 🏆';
        if (averageRating >= 4.0) return 'Хорошее качество 👍';
        if (averageRating >= 3.0) return 'Удовлетворительное качество ✅';
        if (averageRating >= 2.0) return 'Требует улучшения ⚠️';
        return 'Низкое качество ❌';
    }
    
    viewAllTrajectories() {
        // Переход к странице со всеми траекториями
        window.location.href = `/all-trajectories/${encodeURIComponent(this.videoFilename)}`;
    }
    
    uploadNewVideo() {
        // Переход к загрузке нового видео
        window.location.href = '/';
    }
    
    exportResults() {
        try {
            const results = {
                video_filename: this.videoFilename,
                total_trajectories: this.totalTrajectories,
                rated_trajectories: this.ratedTrajectories.size,
                average_rating: this.calculateAverageRating().toFixed(1),
                quality_level: this.getQualityLevel(this.calculateAverageRating()),
                completion_date: new Date().toISOString(),
                export_timestamp: new Date().toLocaleString('ru-RU')
            };
            
            // Создаем JSON файл для скачивания
            const dataStr = JSON.stringify(results, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            
            // Создаем ссылку для скачивания
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `aura_results_${this.videoFilename}_${new Date().toISOString().split('T')[0]}.json`;
            
            // Скачиваем файл
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showNotification('📥 Результаты экспортированы!', 'success');
            
        } catch (error) {
            console.error('Ошибка экспорта:', error);
            this.showNotification('❌ Ошибка экспорта результатов', 'error');
        }
    }
    
    showNotification(message, type = 'info') {
        // Создаем уведомление
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Стили для уведомления
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1001;
            animation: slideInRight 0.3s ease;
            max-width: 300px;
        `;
        
        // Цвета для разных типов
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            info: '#17a2b8'
        };
        
        notification.style.background = colors[type] || colors.info;
        
        // Добавляем на страницу
        document.body.appendChild(notification);
        
        // Удаляем через 3 секунды
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    // Методы для инициализации данных
    setVideoInfo(filename, totalTrajectories) {
        this.videoFilename = filename;
        this.totalTrajectories = totalTrajectories;
    }
    
    setCurrentTrajectory(trajectoryId) {
        this.currentTrajectoryId = trajectoryId;
        this.currentRating = 0;
        
        console.log('🎯 Установлена траектория:', {
            trajectoryId: trajectoryId,
            type: typeof trajectoryId,
            currentTrajectoryId: this.currentTrajectoryId
        });
        
        // Сбрасываем звезды
        document.querySelectorAll('.stars i').forEach(star => {
            star.classList.remove('selected', 'active');
        });
        
        // Сбрасываем комментарий
        const comment = document.getElementById('comment');
        if (comment) comment.value = '';
        
        // Деактивируем кнопку отправки
        const submitBtn = document.getElementById('submit-rating');
        if (submitBtn) submitBtn.disabled = true;
        
        // Обновляем текст
        const ratingText = document.querySelector('.rating-text');
        if (ratingText) ratingText.textContent = 'Выберите оценку';
        
        // Обновляем кнопки навигации
        const prevBtn = document.getElementById('prev-trajectory');
        const nextBtn = document.getElementById('next-trajectory');
        
        if (prevBtn) prevBtn.disabled = (trajectoryId <= 0);
        if (nextBtn) nextBtn.disabled = (trajectoryId >= this.totalTrajectories - 1);
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    window.trajectoryRating = new TrajectoryRating();
    
    // Получаем данные из скрытых полей
    const videoFilename = document.getElementById('video-filename')?.value;
    const trajectoryId = document.getElementById('trajectory-id-value')?.value;
    const totalTrajectories = document.getElementById('total-trajectories-value')?.value;
    
    console.log('🔍 Данные для инициализации:', { videoFilename, trajectoryId, totalTrajectories });
    
    if (videoFilename && trajectoryId && totalTrajectories) {
        window.trajectoryRating.setVideoInfo(videoFilename, parseInt(totalTrajectories));
        window.trajectoryRating.setCurrentTrajectory(parseInt(trajectoryId));
        console.log('✅ Данные инициализированы');
    } else {
        console.error('❌ Не удалось получить данные для инициализации');
    }
});

// CSS анимации для уведомлений
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.9); }
    }
`;
document.head.appendChild(style);
