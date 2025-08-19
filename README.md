# Определение качества вина

Датасет Red Wine Quality - [https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data)

## 1. Структура проекта

```Python
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
```


## 2. Выбор моделей, гиперпараметров и метрик

Для обучения выбрала две модели: ***Линейная регрессия* (Lasso и Ridge)** и ***Случайный лес***.

У линейной регрессии перебирала следующие параметры:

```Python
'alpha': [0.1, 0.5, 1],
'fit_intercept': [True, False]
```


У случайного леса перебирала следующие параметры:

```Python
'n_estimators': [50, 100, 200],
'max_depth': [5, 10, 20],
'min_samples_split': [2, 5],
'min_samples_leaf': [1, 2]
```


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

*Метрики округлены до двух знаков после запятой*

|Модель|MAE|MSE|R²|
|-|-|-|-|
|Лин. регрессия (Ридж)|0.50|0.39|0.40|
|Лин. регрессия (Лассо)|0.55|0.50|0.24|
|Случайный лес|**0.43**|**0.31**|**0.53**|

## Вывод

Случайный лес оказался лучше по всем метрикам: MAE и MSE меньше, а R² больше, поэтому эта модель выбрана как итоговая модель для проекта.

## 3. Инструкции по запуску всех компонентов

### Установка зависимостей

1. Клонируйте репозиторий:

  ```Shell
git clone https://github.com/elizaveta812/Vine-Quality-Project.git
cd mlops-project-el-zimina
```


### Настройка DVC

2. Инициализируйте DVC:

  ```Shell
dvc init
```


3. Настройте удаленное хранилище для DVC (в данном случае MinIO):

  ```Shell
dvc remote add -d s3_mlops s3://el-zimina
```


4. Загрузите данные:

  ```Shell
dvc pull
```


### Запуск DAG в Airflow

5. Запустите Airflow:

  ```Shell
airflow db init
airflow webserver --port 8080
airflow scheduler
```


6. Перейдите в веб-интерфейс Airflow по ссылке `http://localhost:8080` и запустите DAG `first_dag`.

### Запуск API

7. Запустите FastAPI приложение:

  ```Shell
uvicorn api:app --host 0.0.0.0 --port 8000
```


8. Перейдите в браузер по ссылке `http://localhost:8000/docs`, чтобы увидеть документацию API и протестировать эндпоинты.

### Запуск тестов

9. Для запуска тестов используйте:

  ```Shell
pytest tests/
```


