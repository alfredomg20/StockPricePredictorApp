import logging
import os
import pytz
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')

# Set timezone to New York Time for aligment with stock market hours
timezone = pytz.timezone('America/New_York')

# Load environment variables from .env file
load_dotenv()

# Define list of required environment variables
REQUIRED_ENV_VARS = [
    'TYPE',
    'PROJECT_ID',
    'DATASET_ID',
    'STOCKS_TABLE_ID',
    'PRIVATE_KEY_ID',
    'PRIVATE_KEY',
    'CLIENT_EMAIL',
    'CLIENT_ID',
    'AUTH_URI',
    'TOKEN_URI',
    'AUTH_PROVIDER_X509_CERT_URL',
    'CLIENT_X509_CERT_URL',
    'UNIVERSE_DOMAIN',
]

# Check if all required environment variables are set
for var in REQUIRED_ENV_VARS:
    if var not in os.environ:
        raise EnvironmentError(f"Required environment variable '{var}' is not set.")

# Allowed domains for CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

# Google Cloud
# Create dictionary with credentials environment variables
CREDENTIALS_DICT = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": os.getenv("PRIVATE_KEY"),
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("UNIVERSE_DOMAIN"),
}
# Google Cloud project ID
PROJECT_ID = os.getenv("PROJECT_ID")
# BigQuery dataset and table ID
DATASET_ID = os.getenv("DATASET_ID")
STOCKS_TABLE_ID = os.getenv("STOCKS_TABLE_ID")
