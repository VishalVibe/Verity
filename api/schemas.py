from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., max_length=255, description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, max_length=128, description="User password")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., max_length=255, description="User email address")
    password: str = Field(..., max_length=128, description="User password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime
    remaining_quota: int

    model_config = ConfigDict(from_attributes=True)


class RunSummary(BaseModel):
    id: int
    provider: str
    stats: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RunDetail(BaseModel):
    id: int
    provider: str
    answer: str
    context: str
    claims: list
    stats: dict
    status: str
    error: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ApiKeyResponse(BaseModel):
    id: int
    name: str
    prefix: str
    created_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreateResponse(BaseModel):
    api_key: str
    id: int
    name: str
    prefix: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DashboardStatsResponse(BaseModel):
    total_runs: int
    average_accuracy: float
    hallucinations_breakdown: dict
    activity_history: list

    model_config = ConfigDict(from_attributes=True)