# MlOps Project El Zimina
# Проект по определению качества вина

## 1. Структура проекта

**mlops-project-el-zimina**

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

Было выбрано две модели: **_Линейная регрессия_ (Lasso и Ridge)** и **_Случайный лес_**.

У линейной регрессии перебирали следующие параметры:

    'alpha': [0.1, 0.5, 1],
    'fit_intercept': [True, False]

У случайного леса перебирали следующие параметры:

    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]

Для метрик качества выбрали **MAE**, **MSE** и **R² Score**.

## Обучение линейной регрессии (Ридж)

**Лучшими гиперпараметрами оказались:**
 {'alpha': 0.5, 'fit_intercept': True}.

**Метрики следующие:** MAE: 0.5048322818079841, MSE: 0.3917764198549262, R²: 0.40050052461715757

## Обучение линейной регрессии (Лассо)

**Лучшими гиперпараметрами оказались:** {'alpha': 0.1, 'fit_intercept': False}.

**Метрики следующие:** MAE: 0.5459815870835096, MSE: 0.49566859042966077, R²: 0.24152387722474544

## Обучение случайного леса

**Лучшими гиперпараметрами оказались:** {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}.

**Метрики следующие:** MAE: 0.42543858108225213, MSE: 0.30648522358528246, R²: 0.5310138093047876

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
   git clone https://git.lab.karpov.courses/el-zimina/mlops-project-el-zimina.git
   cd mlops-project-el-zimina
   ```


### Настройка DVC

1. Инициализируйте DVC:
   ```bash
   dvc init
   ```

2. Настройте удаленное хранилище для DVC (в нашем случае MinIO):
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