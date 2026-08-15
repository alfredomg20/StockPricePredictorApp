from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class APIConfigSchema(BaseModel):
    title: str = Field(
        default="Stock Prediction API",
        description="API title used in OpenAPI documentation."
    )
    description: str = Field(
        default="API for training and predicting stock prices using linear regression.",
        description="API description used in OpenAPI documentation."
    )
    version: str = Field(
        default="0.1.0",
        description="API version string."
    )
    logger_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the application."
    )


class EnvironmentConfigSchema(BaseModel):
    environment: Literal["dev", "prod", "test"] = Field(
        default="dev",
        description="Deployment environment."
    )
    required_env_vars: list[str] = Field(
        default_factory=list,
        description="List of environment variables required for startup validation."
    )
    allowed_origins: list[str] = Field(
        default_factory=list,
        description="List of allowed origins for CORS."
    )
    timezone: Any = Field(
        default="America/New_York",
        description="Timezone object or string used for alignment with stock market hours."
    )


class GCloudCredentialsSchema(BaseModel):
    type: str | None = Field(None, description="Google service-account type.")
    private_key_id: str | None = Field(None, description="GCP private key ID.")
    private_key: str | None = Field(None, description="GCP private key.")
    client_email: str | None = Field(None, description="GCP client email.")
    client_id: str | None = Field(None, description="GCP client ID.")
    auth_uri: str | None = Field(None, description="GCP auth URI.")
    token_uri: str | None = Field(None, description="GCP token URI.")
    auth_provider_x509_cert_url: str | None = Field(
        None, description="GCP auth provider x509 certificate URL."
    )
    client_x509_cert_url: str | None = Field(
        None, description="GCP client x509 certificate URL."
    )
    universe_domain: str | None = Field(
        None, description="GCP universe domain."
    )


class GCloudConfigSchema(BaseModel):
    project_id: str | None = Field(None, description="GCP project ID.")
    dataset_id: str | None = Field(None, description="BigQuery dataset ID.")
    stocks_table_id: str | None = Field(None, description="BigQuery stocks table ID.")
    credentials: GCloudCredentialsSchema = Field(
        default_factory=GCloudCredentialsSchema,
        description="GCP Service Account Credentials JSON schema."
    )


class PathsConfigSchema(BaseModel):
    models_dir: Path = Field(
        default=Path("app/models"),
        description="Global path to store models"
    )
    frontend_dir: Path = Field(
        default=Path("frontend/"),
        description="Global path for frontend static files"
    )


class FullConfigSchema(BaseModel):
    api: APIConfigSchema = Field(
        default_factory=APIConfigSchema,
        description="API metadata configuration."
    )
    env: EnvironmentConfigSchema = Field(
        default_factory=EnvironmentConfigSchema,
        description="Environment configuration."
    )
    gcloud: GCloudConfigSchema = Field(
        default_factory=GCloudConfigSchema,
        description="Google Cloud / BigQuery configuration."
    )
    paths: PathsConfigSchema = Field(
        default_factory=PathsConfigSchema,
        description="Application paths configuration."
    )