from typing import Any, Literal

from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    task_id: str = Field(..., description="jd-backend task id")
    raw_text: str = Field(..., min_length=1, description="Extracted patent raw text")
    user_id: str | None = Field(default=None, description="Optional user identifier")
    user_prefer: Literal["nation", "area"] = Field(default="nation")
    user_prefer_nation: str | None = Field(default="South Korea")
    user_prefer_area: str | None = Field(default=None)


class SubmitJobResponse(BaseModel):
    task_id: str
    status: Literal["queued"]


class JobStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None

