import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI-based Birth Cohort Study API")
ENV = os.getenv("ENV", "dev")


BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
MODELS_DIR = BASE_DIR / "models"

LBW_MODEL_PATH = MODELS_DIR / "lbw_model.pkl"
LBW_META_PATH = MODELS_DIR / "lbw_model.meta.json"
