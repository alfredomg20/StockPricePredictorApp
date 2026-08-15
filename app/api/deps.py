from fastapi import Request

from app.schemas.config import FullConfigSchema


def get_config(request: Request) -> FullConfigSchema:
    """Extract global configuration saved in app's state"""
    return request.app.state.config
