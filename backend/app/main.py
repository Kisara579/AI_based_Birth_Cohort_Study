from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import APP_NAME, ENV
from .schemas import PredictRequest, PredictResponse
from .predictor import predict_lbw
from .catalog import get_outcomes as catalog_outcomes, get_schema as catalog_schema




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
    return catalog_outcomes()

@app.get("/schema/{outcome_id}")
def get_schema(outcome_id: str):
    return catalog_schema(outcome_id)


@app.post("/predict/{outcome_id}", response_model=PredictResponse)
def predict(outcome_id: str, request: PredictRequest):
    if outcome_id == "lbw":
        probability, risk_label = predict_lbw(request.data)
        return {"probability": probability, "risk_label": risk_label}

    return {"probability": 0.0, "risk_label": "Not Implemented"}
