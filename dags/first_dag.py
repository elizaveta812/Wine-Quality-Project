import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pickle 


# загружаю данные
def load_data():
    import pandas as pd
    print("Загрузили данные о красном вине")
    return pd.read_csv('data/winequality-red.csv')


# обучаю модель
def train_model(data):
    print("Представим, что тут обучилась модель...")
    return "path/to/model.pkl"


# сохраняю модель
def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)  
    print(f"Модель сохранена по пути: {model_path}")



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
    op_kwargs={'data': load_data_task.output},
    dag=dag,
)

save_model_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model,
    op_kwargs={'model_path': 'path/to/model.pkl'}, 
    dag=dag,
)

# порядок выполнения тасков
load_data_task >> train_model_task >> save_model_task

