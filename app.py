from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI()
# Load the trained model
model = joblib.load("model.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_wine_class(request: Request):
    data = await request.json()

    try:
        features = [
            data["alcohol"],
            data["malic_acid"],
            data["ash"],
            data["alcalinity_of_ash"],
            data["magnesium"],
            data["total_phenols"],
            data["flavanoids"],
            data["nonflavanoid_phenols"],
            data["proanthocyanins"],
            data["color_intensity"],
            data["hue"],
            data["od280_od315_of_diluted_wines"],
            data["proline"]
        ]
    except KeyError as e:
        return {"error": f"Missing field: {e.args[0]}"}

    prediction = model.predict([features])[0]

    return {
        "input": data,
        "predicted_class": int(prediction)
    }
