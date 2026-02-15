import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "PeriPredict API")
ENV = os.getenv("ENV", "dev")
