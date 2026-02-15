from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import APP_NAME, ENV

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

@app.get("/outcomes")
def get_outcomes():
    return[
    {
        "id":"lbw",
        "name":"Low Birth Weight",
        "short":"LBW",
        "description":"Predict risk of low birth weight based on maternal factors.",
    }]
