from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import APP_NAME, ENV
from pydantic import BaseModel
from typing import Dict , Any


app = FastAPI(title=APP_NAME)

# allow local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "env": ENV}

# Example endpoint to return available outcomes for prediction
@app.get("/outcomes")
def get_outcomes():
    return[
    {
        "id":"lbw",
        "name":"Low Birth Weight",
        "short":"LBW",
        "description":"Predict risk of low birth weight based on maternal factors.",
    }]


@app.get("/schema/{outcome_id}")
def get_schema(outcome_id: str):
    if outcome_id == "lbw":
        return{
            "title":"Low Birth Weight Inputs",
            "sections":[
                {
                    "id":"maternal",
                    "title": "Maternal Information",
                    "fields": [
                        {
                            "name": "mage",
                            "label": "Mother age (years)",
                            "type": "number",
                            "min": 10,
                            "max": 60,
                            "required": True,
                            "placeholder": "e.g., 28",
                        },
                        {
                            "name": "meduc",
                            "label": "Mother education (years)",
                            "type": "number",
                            "min": 0,
                            "max": 25,
                            "required": True,
                            "placeholder": "e.g., 12",
                        },
                        {
                            "name": "cigs",
                            "label": "Cigarettes per day",
                            "type": "number",
                            "min": 0,
                            "max": 60,
                            "required": True,
                            "placeholder": "e.g., 0",
                        },
                        {
                            "name": "drink",
                            "label": "Alcohol use",
                            "type": "select",
                            "required": True,
                            "options": [
                                {"value": 0, "label": "No"},
                                {"value": 1, "label": "Yes"},
                            ],
                        },
                    ],
                },
                {
                    "id": "pregnancy",
                    "title": "Pregnancy Care",
                    "fields": [
                        {
                            "name": "monpre",
                            "label": "Month prenatal care started",
                            "type": "number",
                            "min": 1,
                            "max": 9,
                            "required": True,
                            "placeholder": "e.g., 2",
                        },
                        {
                             "name": "npvis",
                             "label": "Number of prenatal visits",
                            "type": "number",
                            "min": 0,
                            "max": 40,
                            "required": True,
                            "placeholder": "e.g., 8",
                        },
                        {
                            "name": "male",
                            "label": "Baby sex",
                            "type": "select",
                            "required": True,
                            "options": 
                            [
                                {"value": 1, "label": "Male"},
                                {"value": 0, "label": "Female"},
                            ],
                        },
                    ],
                },
            ],
        }
    return {"error": "Schema not found"}


class PredictRequest(BaseModel):
    data: Dict[str, Any]


@app.post("/predict/{outcome_id}")
def predict(outcome_id: str, request: PredictRequest):
    if outcome_id == "lbw":
        values = request.data

        score = 0

        if values.get("mage") is not None and values["mage"] < 18:
            score += 0.12

        if values.get("meduc") is not None and values["meduc"] < 10:
            score += 0.08

        if values.get("cigs") is not None:
            score += min(0.25, values["cigs"] * 0.01)

        if values.get("drink") == 1:
            score += 0.12

        if values.get("monpre") is not None and values["monpre"] > 3:
            score += 0.10

        if values.get("npvis") is not None and values["npvis"] < 4:
            score += 0.10

        probability = max(0.05, min(0.95, 0.15 + score))
        risk_label = "High Risk" if probability >= 0.5 else "Low Risk"

        return {
            "probability": probability,
            "risk_label": risk_label,
        }

    return {"error": "Prediction not implemented for this outcome"}