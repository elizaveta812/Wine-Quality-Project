# Определение качества вина

Датасет Red Wine Quality - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data

## 1. Структура проекта

    .dvc
        cache
        tmp
        .gitignore
        config
    dags
        .gitkeep
        first_dag.py
    data
        .gitignore
        winequality-red.csv.dvc
    models
        .gitkeep
        random_forest_model.pkl.dvc
    src
        api
            app.py
    tests
        test_api.py
    .dvcignore
    .gitignore
    .gitlab-ci.yml
    README.md


## 2. Выбор моделей, гиперпараметров и метрик

Для обучения выбрала две модели: **_Линейная регрессия_ (Lasso и Ridge)** и **_Случайный лес_**.

У линейной регрессии перебирала следующие параметры:

    'alpha': [0.1, 0.5, 1],
    'fit_intercept': [True, False]

У случайного леса перебирала следующие параметры:

    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]

Для метрик качества выбрала **MAE**, **MSE** и **R² Score**.

## Обучение линейной регрессии (Ридж)

**Лучшими гиперпараметрами оказались:**
 {'alpha': 0.5, 'fit_intercept': True}.

**Метрики следующие:** MAE: 0.5048, MSE: 0.3918, R²: 0.4005

## Обучение линейной регрессии (Лассо)

**Лучшими гиперпараметрами оказались:** {'alpha': 0.1, 'fit_intercept': False}.

**Метрики следующие:** MAE: 0.5459, MSE: 0.4957, R²: 0.2415

## Обучение случайного леса

**Лучшими гиперпараметрами оказались:** {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}.

**Метрики следующие:** MAE: 0.4254, MSE: 0.3065, R²: 0.5310

## Сводная таблица с результатами

_Метрики округлены до двух знаков после запятой_


| Модель                | MAE    | MSE    |R²     |
| ------                | ------ | ------ |------ |
| Лин. регрессия (Ридж) |  0.50  |0.39    |0.40   |
| Лин. регрессия (Лассо)|  0.55  |0.50    |0.24   |
| Случайный лес         |**0.43**|**0.31**|**0.53**|
## Вывод

Случайный лес оказался лучше по всем метрикам: MAE и MSE меньше, а R² больше, поэтому эта модель выбрана как итоговая модель для проекта.

## 3. Инструкции по запуску всех компонентов

### Установка зависимостей

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/elizaveta812/Vine-Quality-Project.git
   cd mlops-project-el-zimina
   ```


### Настройка DVC

1. Инициализируйте DVC:
   ```bash
   dvc init
   ```

2. Настройте удаленное хранилище для DVC (в данном случае MinIO):
   ```bash
   dvc remote add -d s3_mlops s3://el-zimina
   ```

3. Загрузите данные:
   ```bash
   dvc pull
   ```

### Запуск DAG в Airflow

1. Запустите Airflow:
   ```bash
   airflow db init
   airflow webserver --port 8080
   airflow scheduler
   ```

3. Перейдите в веб-интерфейс Airflow по ссылке `http://localhost:8080` и запустите DAG `first_dag`.

### Запуск API

1. Запустите FastAPI приложение:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. Перейдите в браузер по ссылке `http://localhost:8000/docs`, чтобы увидеть документацию API и протестировать эндпоинты.

### Запуск тестов

1. Для запуска тестов используйте:
   ```bash
   pytest tests/
   ```
