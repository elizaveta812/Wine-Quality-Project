from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import boto3
import os

app = FastAPI()

# настройка клиента минё
s3_client = boto3.client('s3',
                          endpoint_url='https://s3.lab.karpov.courses',
                          aws_access_key_id='FLQ92JctrzqXRKchZbqm',
                          aws_secret_access_key='fLePd1XLNkin1Qgm1OFZwPieANSWs0NG5uBUgCSk')

# параметры для загрузки модели
bucket_name = 'el-zimina'
model_key = 'models/wine_quality_model.pkl'
local_model_path = '/tmp/wine_quality_model.pkl'


# загружаю модель из минё
def load_model():
    try:
        s3_client.download_file(bucket_name, model_key, local_model_path)
        with open(local_model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {e}")


model = load_model()


# валидация входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.post("/predict")
def predict(data: WineFeatures) -> float:
    try:
        features = [[
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol
        ]]
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении предсказания: {e}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "model": "Random Forest",
        "version": "1.0",
        "metrics": {
            "MAE": 0.43,
            "MSE": 0.31,
            "R2": 0.53
        }
    }

