from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import boto3
import os

app = FastAPI()


# получаю переменные окружения
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# настройка клиента минё
s3_client = boto3.client('s3',
                          endpoint_url='https://s3.lab.karpov.courses',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)

# параметры для загрузки модели
bucket_name = 'el-zimina'
model_key = 'models/wine_quality_model.pkl'
local_model_path = '/tmp/wine_quality_model.pkl'

model = None


# загружаю модель из минё
async def load_model():
    global model
    try:
        s3_client.download_file(bucket_name, model_key, local_model_path)
        with open(local_model_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {e}")


async def unload_model():
    global model
    model = None


@app.on_event("startup")
async def startup_event():
    await load_model()


@app.on_event("shutdown")
async def shutdown_event():
    await unload_model()


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
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

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

