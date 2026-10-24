from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


ModelType = Literal["land-cover", "change-detection"]
VersionStatus = Literal["UPLOADING", "BUILDING", "VALIDATING", "AVAILABLE", "BUILD_FAILED", "VALIDATION_FAILED", "OFFLINE"]
Tag = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]


class ModelResourceSummary(BaseModel):
    # A version being uploaded may not have resource metadata yet. Only these
    # public fields can leave the API; other manifest keys stay private.
    cpu_cores: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    gpu_count: int | None = Field(default=None, ge=0)


class ModelVersionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: str
    status: VersionStatus
    framework: str
    input_description: str
    input_bands: list[int]
    resource_requirements: ModelResourceSummary


class MLModelListItem(BaseModel):
    id: int
    model_name: str
    model_type: ModelType
    description: str | None
    tags: list[str]
    production_version_id: int | None
    version: ModelVersionSummary | None
    business_capabilities: list[Literal["land-cover"]]
    updated_at: datetime


class MLModelListResponse(BaseModel):
    items: list[MLModelListItem]
    total: int
    page: int
    pageSize: int
    can_manage: bool


class MLModelUpdateRequest(BaseModel):
    """Only catalog metadata can be edited; asset and version fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=40)]
    description: str | None = Field(max_length=200)
    tags: list[Tag] = Field(max_length=20)
