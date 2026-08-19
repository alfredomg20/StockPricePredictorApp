from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_config
from app.api.health import ready_cache
from app.main import app


@pytest.fixture
def mock_config():
    """Minimal config double exposing only what the health router reads"""
    config = MagicMock()
    config.api.version = "0.2.0"
    config.env.environment = "test"
    config.env.required_env_vars = ["PROJECT_ID", "DATASET_ID"]
    config.paths.models_dir = "tmp/models"
    return config


@pytest.fixture
def client(mock_config):
    """TestClient with get_config overridden and the readiness cache cleared per test"""
    app.dependency_overrides[get_config] = lambda: mock_config
    ready_cache.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mock_models_dir_ok():
    """Patch Path so the models directory check reports as existing"""
    with patch("app.api.health.Path") as mock:
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        mock.return_value = mock_dir
        yield mock


class TestHealthRouter:
    def test_live_returns_alive(self, client):
        """Test liveness probe returns alive without checking any dependency"""
        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    @patch("app.api.health.validate_env_variables")
    def test_ready_success(self, mock_validate, client, mock_models_dir_ok):
        """Test readiness returns 200 when env vars and models dir are both OK"""
        mock_validate.return_value = []

        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"] == {"env_vars": True, "models_dir": True}

    @patch("app.api.health.validate_env_variables")
    def test_ready_missing_env_vars(self, mock_validate, client, mock_models_dir_ok):
        """Test readiness returns 503 when required env vars are missing"""
        mock_validate.return_value = ["PROJECT_ID"]

        response = client.get("/api/v1/health/ready")

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["status"] == "not_ready"
        assert detail["checks"]["env_vars"] is False

    @patch("app.api.health.validate_env_variables")
    def test_ready_models_dir_missing(self, mock_validate, client):
        """Test readiness returns 503 when the models directory is not accessible"""
        mock_validate.return_value = []
        with patch("app.api.health.Path") as mock_path:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = False
            mock_path.return_value = mock_dir

            response = client.get("/api/v1/health/ready")

        assert response.status_code == 503
        assert response.json()["detail"]["checks"]["models_dir"] is False

    @patch("app.api.health.validate_env_variables")
    def test_ready_uses_cache_on_second_call(self, mock_validate, client, mock_models_dir_ok):
        """Test that a second call within the TTL window skips re-running the checks"""
        mock_validate.return_value = []

        first = client.get("/api/v1/health/ready")
        second = client.get("/api/v1/health/ready")

        assert first.status_code == 200
        assert second.status_code == 200
        mock_validate.assert_called_once()

    def test_info_returns_version_environment_and_uptime(self, client, mock_config):
        """Test info endpoint returns version, environment and a non-negative uptime"""
        response = client.get("/api/v1/health/info")

        assert response.status_code == 200
        body = response.json()
        assert body["version"] == mock_config.api.version
        assert body["environment"] == mock_config.env.environment
        assert body["uptime_seconds"] >= 0