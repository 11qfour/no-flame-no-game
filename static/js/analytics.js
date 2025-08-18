// Скрипт для страницы аналитики

document.addEventListener('DOMContentLoaded', function() {
    // Получаем video_id из URL
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('video_id') || 'demo';
    
    // Загружаем данные аналитики
    loadAnalyticsData(videoId);
});

async function loadAnalyticsData(videoId) {
    try {
        // В реальном приложении здесь был бы API-вызов
        // Для демо используем заглушечные данные
        const analyticsData = generateDemoData();
        
        // Заполняем интерфейс данными
        populateAnalytics(analyticsData);
        
    } catch (error) {
        console.error('Ошибка загрузки аналитики:', error);
        showError('Не удалось загрузить данные аналитики');
    }
}

function generateDemoData() {
    // Генерируем демонстрационные данные для презентации
    return {
        summary: {
            total_visitors: Math.floor(Math.random() * 50) + 20,
            max_concurrent_visitors: Math.floor(Math.random() * 15) + 5,
            avg_concurrent_visitors: Math.floor(Math.random() * 8) + 3,
            avg_visit_duration: Math.floor(Math.random() * 180) + 60,
            video_duration: 3600,
            peak_time: `${Math.floor(Math.random() * 12) + 10}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`
        },
        heatmap: {
            image_path: '/static/images/demo_heatmap.png',
            hot_spots: [
                { rank: 1, intensity: 0.95, x: 320, y: 180 },
                { rank: 2, intensity: 0.78, x: 150, y: 220 },
                { rank: 3, intensity: 0.65, x: 480, y: 160 }
            ]
        },
        desire_paths: {
            image_path: '/static/images/demo_paths.png',
            total_paths: Math.floor(Math.random() * 40) + 15,
            avg_path_duration: Math.floor(Math.random() * 120) + 30,
            common_patterns: [
                { type: 'popular_entry', location: 'entrance', count: 15, percentage: 75 },
                { type: 'popular_exit', location: 'checkout', count: 12, percentage: 60 }
            ]
        },
        queue_analysis: {
            image_path: '/static/images/demo_queue.png',
            max_concurrent: Math.floor(Math.random() * 15) + 5,
            avg_concurrent: Math.floor(Math.random() * 8) + 3,
            peak_periods: [
                { start_time: '11:30', end_time: '12:15', duration: 45, max_people: 12 },
                { start_time: '14:20', end_time: '14:45', duration: 25, max_people: 8 }
            ]
        },
        anomalies: {
            total_anomalies: Math.floor(Math.random() * 5) + 1,
            anomalies: [
                {
                    type: 'stationary_person',
                    severity: 'medium',
                    description: 'Посетитель стоял у входа в течение 3 минут, возможно затрудняя проход',
                    timestamp: '11:45',
                    person_id: 'P123'
                },
                {
                    type: 'crowd_surge',
                    severity: 'high',
                    description: 'Неожиданное скопление 8 человек возле кассы в 14:30',
                    timestamp: '14:30',
                    people_count: 8
                }
            ],
            severity_breakdown: { low: 1, medium: 2, high: 1 }
        },
        time_analysis: {
            intervals: [
                { time: '10:00', avg_people: 2.5, max_people: 4, activity_level: 'Низкая' },
                { time: '11:00', avg_people: 5.2, max_people: 8, activity_level: 'Средняя' },
                { time: '12:00', avg_people: 8.1, max_people: 12, activity_level: 'Высокая' },
                { time: '13:00', avg_people: 6.8, max_people: 10, activity_level: 'Средняя' },
                { time: '14:00', avg_people: 9.2, max_people: 15, activity_level: 'Очень высокая' }
            ],
            busiest_minute: { time: '12:30', max_people: 15 },
            quietest_minute: { time: '10:15', avg_people: 1.2 }
        }
    };
}

function populateAnalytics(data) {
    // Заполняем сводку
    populateSummary(data.summary);
    
    // Заполняем тепловую карту
    populateHeatmap(data.heatmap);
    
    // Заполняем тропы желаний
    populateDesirePaths(data.desire_paths);
    
    // Заполняем анализ очередей
    populateQueueAnalysis(data.queue_analysis);
    
    // Заполняем аномалии
    populateAnomalies(data.anomalies);
    
    // Заполняем временной анализ
    populateTimeAnalysis(data.time_analysis);
    
    // Заполняем анализ времени пребывания
    if (data.dwell_time_analysis) {
        populateDwellTimeAnalysis(data.dwell_time_analysis);
    }
}

function populateSummary(summary) {
    document.getElementById('totalVisitors').textContent = summary.total_visitors || '-';
    document.getElementById('maxConcurrent').textContent = summary.max_concurrent_visitors || '-';
    document.getElementById('avgDuration').textContent = formatDuration(summary.avg_visit_duration);
    document.getElementById('peakTime').textContent = summary.peak_time || '-';
}

function populateHeatmap(heatmap) {
    const heatmapImage = document.getElementById('heatmapImage');
    const heatmapInsights = document.getElementById('heatmapInsights');
    
    if (heatmap.image_path) {
        heatmapImage.src = heatmap.image_path;
        heatmapImage.onerror = function() {
            this.src = '/static/images/placeholder_heatmap.png';
        };
    }
    
    let insights = [];
    if (heatmap.hot_spots && heatmap.hot_spots.length > 0) {
        insights.push(`Обнаружено ${heatmap.hot_spots.length} зон повышенной активности`);
        insights.push(`Самая популярная зона имеет интенсивность ${Math.round(heatmap.hot_spots[0].intensity * 100)}%`);
        insights.push('Рекомендуется разместить ключевые товары в зонах высокой активности');
    } else {
        insights.push('Равномерное распределение активности по всему пространству');
    }
    
    heatmapInsights.innerHTML = insights.map(insight => `<li>${insight}</li>`).join('');
}

function populateDesirePaths(desirePaths) {
    const pathsImage = document.getElementById('pathsImage');
    const pathsInsights = document.getElementById('pathsInsights');
    
    if (desirePaths.image_path) {
        pathsImage.src = desirePaths.image_path;
        pathsImage.onerror = function() {
            this.src = '/static/images/placeholder_paths.png';
        };
    }
    
    let insights = [];
    insights.push(`Проанализировано ${desirePaths.total_paths} уникальных маршрутов`);
    insights.push(`Средняя длительность перемещения: ${formatDuration(desirePaths.avg_path_duration)}`);
    
    if (desirePaths.common_patterns) {
        const entryPatterns = desirePaths.common_patterns.filter(p => p.type === 'popular_entry');
        const exitPatterns = desirePaths.common_patterns.filter(p => p.type === 'popular_exit');
        
        if (entryPatterns.length > 0) {
            insights.push(`${entryPatterns[0].percentage}% посетителей используют основной вход`);
        }
        if (exitPatterns.length > 0) {
            insights.push(`${exitPatterns[0].percentage}% посетителей выходят через кассу`);
        }
    }
    
    pathsInsights.innerHTML = insights.map(insight => `<li>${insight}</li>`).join('');
}

function populateQueueAnalysis(queueAnalysis) {
    const queueImage = document.getElementById('queueImage');
    const queueInsights = document.getElementById('queueInsights');
    
    if (queueAnalysis.image_path) {
        queueImage.src = queueAnalysis.image_path;
        queueImage.onerror = function() {
            this.src = '/static/images/placeholder_queue.png';
        };
    }
    
    let insights = [];
    insights.push(`Максимальная одновременная загруженность: ${queueAnalysis.max_concurrent} человек`);
    insights.push(`Средняя загруженность: ${queueAnalysis.avg_concurrent} человек`);
    
    if (queueAnalysis.peak_periods && queueAnalysis.peak_periods.length > 0) {
        const peak = queueAnalysis.peak_periods[0];
        insights.push(`Главный пик активности: ${peak.start_time} - ${peak.end_time} (${peak.max_people} человек)`);
        insights.push('Рекомендуется усилить персонал в пиковые часы');
    }
    
    if (queueAnalysis.avg_concurrent > queueAnalysis.max_concurrent * 0.7) {
        insights.push('⚠️ Высокая постоянная загруженность - рассмотрите расширение пространства');
    }
    
    queueInsights.innerHTML = insights.map(insight => `<li>${insight}</li>`).join('');
}

function populateAnomalies(anomalies) {
    const anomaliesContainer = document.getElementById('anomaliesContainer');
    
    if (!anomalies.anomalies || anomalies.anomalies.length === 0) {
        anomaliesContainer.innerHTML = `
            <div class="no-anomalies">
                <i class="fas fa-check-circle" style="color: #28a745; font-size: 3em; margin-bottom: 15px;"></i>
                <h4>Аномалий не обнаружено</h4>
                <p>Поведение посетителей соответствует нормальным паттернам</p>
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="anomalies-summary">
            <h4>Обнаружено аномалий: ${anomalies.total_anomalies}</h4>
            <div class="severity-breakdown">
                <span class="severity-badge low">Низкая: ${anomalies.severity_breakdown?.low || 0}</span>
                <span class="severity-badge medium">Средняя: ${anomalies.severity_breakdown?.medium || 0}</span>
                <span class="severity-badge high">Высокая: ${anomalies.severity_breakdown?.high || 0}</span>
            </div>
        </div>
        <div class="anomalies-list">
    `;
    
    anomalies.anomalies.forEach(anomaly => {
        html += `
            <div class="anomaly-item severity-${anomaly.severity}">
                <div class="anomaly-header">
                    <div class="anomaly-icon">
                        ${getAnomalyIcon(anomaly.type)}
                    </div>
                    <div class="anomaly-title">
                        ${getAnomalyTitle(anomaly.type)}
                    </div>
                </div>
                <div class="anomaly-description">
                    ${anomaly.description}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    anomaliesContainer.innerHTML = html;
}

function populateTimeAnalysis(timeAnalysis) {
    const timeAnalysisContainer = document.getElementById('timeAnalysisContainer');
    
    if (!timeAnalysis.intervals || timeAnalysis.intervals.length === 0) {
        timeAnalysisContainer.innerHTML = '<p>Недостаточно данных для временного анализа</p>';
        return;
    }
    
    let html = `
        <div class="time-summary">
            <div class="time-highlight">
                <strong>Самое загруженное время:</strong> ${timeAnalysis.busiest_minute?.time || 'N/A'} 
                (${timeAnalysis.busiest_minute?.max_people || 0} человек)
            </div>
            <div class="time-highlight">
                <strong>Самое спокойное время:</strong> ${timeAnalysis.quietest_minute?.time || 'N/A'} 
                (${timeAnalysis.quietest_minute?.avg_people || 0} человек)
            </div>
        </div>
        <div class="time-intervals">
            <h4>Активность по часам:</h4>
    `;
    
    timeAnalysis.intervals.forEach(interval => {
        html += `
            <div class="time-interval">
                <div class="time-label">${interval.time}</div>
                <div class="time-stats">
                    <span>Среднее: ${interval.avg_people}</span>
                    <span>Максимум: ${interval.max_people}</span>
                </div>
                <div class="activity-level ${getActivityClass(interval.activity_level)}">
                    ${interval.activity_level}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    timeAnalysisContainer.innerHTML = html;
}

// Вспомогательные функции
function formatDuration(seconds) {
    if (!seconds) return '-';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes > 0) {
        return `${minutes}м ${remainingSeconds}с`;
    } else {
        return `${remainingSeconds}с`;
    }
}

function getAnomalyIcon(type) {
    const icons = {
        'stationary_person': '<i class="fas fa-user-clock"></i>',
        'crowd_surge': '<i class="fas fa-users"></i>',
        'long_visit': '<i class="fas fa-hourglass-half"></i>'
    };
    return icons[type] || '<i class="fas fa-exclamation"></i>';
}

function getAnomalyTitle(type) {
    const titles = {
        'stationary_person': 'Блокировка прохода',
        'crowd_surge': 'Скопление людей',
        'long_visit': 'Необычно долгое посещение'
    };
    return titles[type] || 'Аномалия';
}

function getActivityClass(activityLevel) {
    const classes = {
        'Низкая': 'low',
        'Средняя': 'medium',
        'Высокая': 'high',
        'Очень высокая': 'very-high'
    };
    return classes[activityLevel] || 'low';
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-alert';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>${message}</span>
    `;
    
    document.querySelector('.analytics-content').prepend(errorDiv);
    
    setTimeout(() => errorDiv.remove(), 5000);
}

// Дополнительные стили для страницы аналитики
const analyticsStyles = `
    <style>
        .no-anomalies {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        
        .anomalies-summary {
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 10px;
        }
        
        .severity-breakdown {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .severity-badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
            color: white;
        }
        
        .severity-badge.low { background: #28a745; }
        .severity-badge.medium { background: #ffc107; }
        .severity-badge.high { background: #dc3545; }
        
        .time-summary {
            background: rgba(102, 126, 234, 0.05);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
        }
        
        .time-highlight {
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .time-highlight:last-child {
            margin-bottom: 0;
        }
        
        .time-intervals h4 {
            margin-bottom: 20px;
            color: #333;
        }
        
        .time-stats {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #666;
        }
        
        .error-alert {
            background: #f8d7da;
            color: #721c24;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            border: 1px solid #f5c6cb;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        @media (max-width: 768px) {
            .severity-breakdown {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .time-stats {
                flex-direction: column;
                gap: 5px;
            }
        }
    </style>
`;

document.head.insertAdjacentHTML('beforeend', analyticsStyles);

function populateDwellTimeAnalysis(dwellData) {
    // Показываем секцию анализа времени пребывания
    const dwellSection = document.getElementById('dwellTimeSection');
    if (dwellSection) {
        dwellSection.style.display = 'block';
    }
    
    // Заполняем список активных зон
    const activeZonesList = document.getElementById('activeZonesList');
    if (activeZonesList && dwellData.active_zones) {
        if (dwellData.active_zones.length > 0) {
            const zonesHTML = dwellData.active_zones.map((zone, index) => {
                // Определяем цвет зоны на основе уровня теплоты
                const heatClass = zone.heat_level || 'none';
                const heatColor = getHeatLevelColor(heatClass);
                
                return `
                    <div class="zone-item ${heatClass}">
                        <div class="zone-info">
                            <div class="zone-location">
                                <span class="zone-color ${heatClass}"></span>
                                Зона ${index + 1}
                            </div>
                            <div class="zone-description">${zone.description}</div>
                            <div class="zone-coordinates">
                                📍 Координаты: (${Math.round(zone.x)}, ${Math.round(zone.y)})
                            </div>
                        </div>
                        <div class="zone-time">
                            <strong>${zone.dwell_time.toFixed(1)}с</strong>
                            <div class="heat-level">${getHeatLevelLabel(heatClass)}</div>
                        </div>
                    </div>
                `;
            }).join('');
            
            activeZonesList.innerHTML = zonesHTML;
        } else {
            activeZonesList.innerHTML = '<p>Активные зоны не обнаружены</p>';
        }
    }
    
    // Обновляем информацию о порогах
    if (dwellData.time_thresholds) {
        console.log('🔥 Пороги времени:', dwellData.time_thresholds);
    }
    
    console.log('🔥 Анализ времени пребывания загружен:', dwellData);
}

function getHeatLevelColor(heatLevel) {
    const colors = {
        'light': '#ffa726',
        'medium': '#ffb74d', 
        'high': '#ff7043',
        'very_high': '#d84315',
        'none': '#9e9e9e'
    };
    return colors[heatLevel] || '#9e9e9e';
}

function getHeatLevelLabel(heatLevel) {
    const labels = {
        'light': 'Легкая',
        'medium': 'Средняя',
        'high': 'Высокая', 
        'very_high': 'Очень высокая',
        'none': 'Минимальная'
    };
    return labels[heatLevel] || 'Неизвестно';
}
