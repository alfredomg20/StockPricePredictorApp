import logging
import time
from pathlib import Path

from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_config
from app.schemas.config import FullConfigSchema
from app.schemas.health import InfoResponse, ReadyResponse
from app.utils.validation_utils import validate_env_variables

logger = logging.getLogger('app')

router = APIRouter(prefix="/health", tags=["health"])

start_time = time.time()

# Configure caching
TIME_TO_LIVE = 15
ready_cache = TTLCache(maxsize=1, ttl=TIME_TO_LIVE)

@router.get("/live", status_code=status.HTTP_200_OK)
async def live():
    """Liveness probe. Confirms the process is running; no external dependency checks."""
    return {"status": "alive"}

@router.get("/ready", response_model=ReadyResponse)
async def ready(config: FullConfigSchema = Depends(get_config)):
    """Readiness probe. Verifies required env vars are set and the models directory is accessible."""
    if "result" in ready_cache:
        cached = ready_cache["result"]
        if cached.status != "ready":
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=cached.dict())
        return cached

    checks = {}

    missing_vars = validate_env_variables(config.env.required_env_vars)
    checks["env_vars"] = not missing_vars
    if missing_vars:
        logger.warning(f"Readiness check: missing env vars: {', '.join(missing_vars)}")

    models_path = Path(config.paths.models_dir)
    checks["models_dir"] = models_path.exists() and models_path.is_dir()

    result = ReadyResponse(
        status="ready" if all(checks.values()) else "not_ready",
        checks=checks,
    )
    ready_cache["result"] = result

    if result.status != "ready":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.dict())
    return result

@router.get("/info", response_model=InfoResponse)
async def info(config: FullConfigSchema = Depends(get_config)):
    """Returns non-sensitive service info """
    return InfoResponse(
        version=config.api.version,
        environment=config.env.environment,
        uptime_seconds=round(time.time() - start_time, 2),
    )