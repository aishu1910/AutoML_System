from fastapi import FastAPI
import joblib
import numpy as np
from schema import InputData
app = FastAPI()

# Load model once when server starts
model = joblib.load("best_model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API is running 🚀"}


# Example prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4
    ]).reshape(1, -1)

    prediction = model.predict(features)

    return {"prediction": prediction.tolist()}