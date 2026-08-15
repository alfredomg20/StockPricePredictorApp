import os
from pathlib import Path

import pytz
from dotenv import load_dotenv

from app.schemas.config import FullConfigSchema

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

CONFIG = FullConfigSchema(
    api={
        "title": "Stock Prediction API",
        "description": "API for training and predicting stock prices using linear regression.",
        "version": "0.1.2",
        "logger_level": "INFO",
    },
    env={
        "environment": os.getenv("ENVIRONMENT", "dev"),
        "required_env_vars": [
            "TYPE",
            "PROJECT_ID",
            "DATASET_ID",
            "STOCKS_TABLE_ID",
            "PRIVATE_KEY_ID",
            "PRIVATE_KEY",
            "CLIENT_EMAIL",
            "CLIENT_ID",
            "AUTH_URI",
            "TOKEN_URI",
            "AUTH_PROVIDER_X509_CERT_URL",
            "CLIENT_X509_CERT_URL",
            "UNIVERSE_DOMAIN",
        ],
        "allowed_origins": [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()],
        "timezone": pytz.timezone("America/New_York"),
    },
    gcloud={
        "project_id": os.getenv("PROJECT_ID"),
        "dataset_id": os.getenv("DATASET_ID"),
        "stocks_table_id": os.getenv("STOCKS_TABLE_ID"),
        "credentials": {
            "type": os.getenv("TYPE"),
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": os.getenv("PRIVATE_KEY"),
            "client_email": os.getenv("CLIENT_EMAIL"),
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": os.getenv("AUTH_URI"),
            "token_uri": os.getenv("TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
            "universe_domain": os.getenv("UNIVERSE_DOMAIN"),
        },
    },
    paths={
        "models_dir": BASE_DIR / "app" / "models",
        "frontend_dir": BASE_DIR / "frontend",
    },
)