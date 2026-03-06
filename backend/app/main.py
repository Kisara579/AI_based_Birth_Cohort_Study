from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import json

from .config import APP_NAME, ENV, model_paths
from .schemas import PredictRequest, PredictResponse
from .catalog import get_outcomes as catalog_outcomes, get_schema as catalog_schema
from .catalog import get_model_key
from .predictor import predict_with_model_key




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
def outcomes():
    return catalog_outcomes()

@app.get("/schema/{outcome_id}")
def schema(outcome_id: str):
    s = catalog_schema(outcome_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Schema not found")
    return s


# manual model selection
@app.post("/predict/{outcome_id}", response_model=PredictResponse)
def predict(outcome_id: str, request: PredictRequest, model: str | None = Query(default=None)):
    model_key = get_model_key(outcome_id)
    if model_key is None:
        raise HTTPException(status_code=404, detail="Unknown outcome")

    probability, risk_label = predict_with_model_key(model_key, request.data, algorithm=model)
    return {"probability": probability, "risk_label": risk_label}

@app.get("/metrics/{outcome_id}")
def metrics(outcome_id: str, model: str | None = Query(default=None)):
    model_key = get_model_key(outcome_id)
    if model_key is None:
        raise HTTPException(status_code=404, detail="Unknown outcome")
    
    _, _, metrics_path = model_paths(model_key, algorithm=model)

    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train model first.")

    return json.loads(metrics_path.read_text(encoding="utf-8"))