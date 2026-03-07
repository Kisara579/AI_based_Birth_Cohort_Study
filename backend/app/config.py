import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI-based Birth Cohort Study API")
ENV = os.getenv("ENV", "dev")


BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
MODELS_DIR = BASE_DIR / "models"

# Maps catalog model_key → subdirectory name under MODELS_DIR
OUTCOME_FOLDER: dict[str, str] = {
    "lbw": "lbw",
    "pd": "pd",
}


def model_paths(model_key: str, algorithm: str | None = None):
    """
    Return (model_path, meta_path, metrics_path) for the given outcome.

    Looks up the outcome folder via OUTCOME_FOLDER, then reads registry.json
    to determine the active algorithm (or uses the supplied ``algorithm``).
    Falls back to the legacy flat layout when no registry is present.
    """
    outcome_folder = OUTCOME_FOLDER.get(model_key, model_key)
    outcome_dir = MODELS_DIR / outcome_folder
    registry_path = outcome_dir / "registry.json"

    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

        resolved_algorithm = algorithm or registry.get("active_model")
        if not resolved_algorithm:
            raise FileNotFoundError(
                f"No active model configured for outcome '{model_key}'."
            )

        algo_dir = outcome_dir / resolved_algorithm

        return (
            algo_dir / "model.pkl",
            algo_dir / "meta.json",
            algo_dir / "metrics.json",
        )

    return (
        MODELS_DIR / f"{model_key}_model.pkl",
        MODELS_DIR / f"{model_key}_model.meta.json",
        MODELS_DIR / f"{model_key}_metrics.json",
    )
