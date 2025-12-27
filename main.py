"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random

# Шаг 1: Улучшенная предобработка данных
def preprocess_data(data):
    # Преобразование даты в формат datetime
    data['dt'] = pd.to_datetime(data['dt'])

    # Извлечение признаков из даты
    data['year'] = data['dt'].dt.year
    data['month'] = data['dt'].dt.month
    data['day_of_week'] = data['dt'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'] >= 5
    data['day_of_year'] = data['dt'].dt.dayofyear
    data['week_of_year'] = data['dt'].dt.isocalendar().week

    # Функция для кодирования категориальных переменных
    categorical_columns = ['management_group_id', 'first_category_id', 'second_category_id',
                           'third_category_id', 'product_id']

    # Кодируем категориальные признаки с помощью OneHotEncoding для улучшенной работы с моделями
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Функция для обработки числовых признаков
    numeric_features = ['n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
                        'holiday_flag', 'activity_flag', 'dow', 'day_of_month', 'week_of_year']

    # Обработка числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Заполнение пропусков
        ('scaler', StandardScaler())  # Стандартизация числовых признаков
    ])

    # Преобразование для всех числовых признаков
    data[numeric_features] = numeric_transformer.fit_transform(data[numeric_features])

    return data

# Функция для вычисления Intersection over Union (IoU)
def calculate_iou(true_lower, true_upper, pred_lower, pred_upper, epsilon=1e-6):
    intersection = max(0, min(true_upper, pred_upper) - max(true_lower, pred_lower))
    union = (true_upper - true_lower) + (pred_upper - pred_lower) - intersection
    return intersection / (union + epsilon)

def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """

    # Фиксируем сид для воспроизводимости
    seed = 322
    np.random.seed(seed)
    random.seed(seed)

    # Загрузка данных
    train_data = pd.read_csv('data/train.csv')
    # Применение функции предобработки данных к тренировочному набору
    processed_train_data = preprocess_data(train_data)

    # Разделение на признаки и целевые переменные
    features = [col for col in processed_train_data.columns if col not in ['price_p05', 'price_p95', 'dt']]
    X = processed_train_data[features]
    y = processed_train_data[['price_p05', 'price_p95']]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Шаг 2: Обучение модели для каждого целевого признака отдельно
    model_p05 = GradientBoostingRegressor(n_estimators=100, random_state=seed)
    model_p95 = GradientBoostingRegressor(n_estimators=100, random_state=seed)

    # Обучение модели для price_p05
    model_p05.fit(X_train, y_train['price_p05'])

    # Обучение модели для price_p95
    model_p95.fit(X_train, y_train['price_p95'])

    # Шаг 3: Получение оценки модели
    y_pred_p05 = model_p05.predict(X_test)
    y_pred_p95 = model_p95.predict(X_test)

    # Оценка модели с использованием IoU
    true_lower = y_test['price_p05'].values
    true_upper = y_test['price_p95'].values
    ious = [calculate_iou(true_lower[i], true_upper[i], y_pred_p05[i], y_pred_p95[i]) for i in range(len(true_lower))]
    mean_iou = np.mean(ious)

    print(f"Mean IoU: {mean_iou}")

    # Шаг 4: Преобразование тестовых данных
    test_data = pd.read_csv('data/test.csv')  # Замените на путь к вашему тестовому набору

    # Применяем ту же обработку, что и для тренировочного набора
    processed_test_data = preprocess_data(test_data)

    # Прогнозирование для тестового набора
    test_predictions_p05 = model_p05.predict(processed_test_data[features])
    test_predictions_p95 = model_p95.predict(processed_test_data[features])

    # Шаг 5: Генерация файла для сабмита
    submission = pd.DataFrame({
        'row_id': test_data.index,  # Используем индекс для уникальных идентификаторов строк
        'price_p05': test_predictions_p05,  # Нижняя граница диапазона
        'price_p95': test_predictions_p95   # Верхняя граница диапазона
    })

    #submission.to_csv('submission.csv', index=False)
    print('Submission saved to submission.csv')

    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


def create_submission(submission):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path

main()