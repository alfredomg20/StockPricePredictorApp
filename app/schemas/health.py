from typing import Literal

from pydantic import BaseModel

ReadyStatus = Literal["ready", "not_ready"]

class ReadyResponse(BaseModel):
    status: ReadyStatus
    checks: dict[str, bool]

class InfoResponse(BaseModel):
    version: str
    environment: str 
    uptime_seconds: float