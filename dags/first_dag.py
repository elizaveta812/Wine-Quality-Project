import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import boto3
from botocore.exceptions import NoCredentialsError
import os


# получаю переменные окружения
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


# настройка клиента минё
s3_client = boto3.client('s3',
                          endpoint_url='https://s3.lab.karpov.courses',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)


# загружаю данные из минё
def load_data():
    try:
        bucket_name = 'el-zimina'
        object_key = 'data/winequality-red.csv'
        local_file_path = '/tmp/winequality-red.csv'

        s3_client.download_file(bucket_name, object_key, local_file_path)

        data = pd.read_csv(local_file_path)
        print("Загрузили данные о красном вине из MinIO")
        return data
    except NoCredentialsError:
        print("Ошибка: Неверные учетные данные для доступа к MinIO")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")


# обучаю модель
def train_model(**kwargs):
    # получаю данные из предыдущего степа
    data = kwargs['ti'].xcom_pull(task_ids='load_data')

    # делю на фичи и таргет
    X = data.drop('quality', axis=1)
    y = data['quality']

    # делю на трейн и на тест
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # обучаю модель
    random_forest_model = RandomForestRegressor(random_state=42,
                                                n_estimators=200,
                                                max_depth=20,
                                                min_samples_split=2,
                                                min_samples_leaf=1)

    random_forest_model.fit(X_train, y_train)

    # предсказания
    rf_predictions = random_forest_model.predict(X_test)

    # вычисляю метрики
    mae_rf = mean_absolute_error(y_test, rf_predictions)
    mse_rf = mean_squared_error(y_test, rf_predictions)
    r2_rf = r2_score(y_test, rf_predictions)

    # логирую результаты
    print("Метрики случайного леса:")
    print(f"MAE: {mae_rf}, MSE: {mse_rf}, R²: {r2_rf}")

    print("Обучили лучшую модель на лучших параметрах")

    return random_forest_model


# сохраняю модель в минё
def save_model(model):
    model_path = 'models/wine_quality_model.pkl'

    # сохраняю модель в локальный файл
    with open('/tmp/wine_quality_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # загружаю в минё
    try:
        s3_client.upload_file('/tmp/wine_quality_model.pkl', 'el-zimina', model_path)
        print(f"Модель сохранена в MinIO по пути: {model_path}")
    except NoCredentialsError:
        print("Ошибка: Неверные учетные данные для доступа к MinIO")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")


# определяю даг
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'first_dag',
    default_args=default_args,
    description='Первый DAG со структурой проекта',
    schedule_interval=timedelta(days=1),
)

# определяю таски
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

save_model_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model,
    op_kwargs={
        'model': '{{ task_instance.xcom_pull(task_ids="train_model") }}'
        },
    dag=dag,
)

# порядок выполнения тасков
load_data_task >> train_model_task >> save_model_task

