"""System catalog; legacy model_library IDs are never reused as catalog IDs."""
from datetime import datetime
from typing import Literal

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint, JSON, String, Text, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column

from models.Base import Base


class SystemModel(Base):
    __tablename__ = "system_models"
    __table_args__ = (
        CheckConstraint("model_type IN ('land-cover', 'change-detection')", name="ck_system_models_type"),
        ForeignKeyConstraint(
            ["id", "production_version_id"],
            ["system_model_versions.model_id", "system_model_versions.id"],
            name="fk_system_models_production_version", use_alter=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255))
    model_type: Mapped[Literal["land-cover", "change-detection"]] = mapped_column(String(32))
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    production_version_id: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SystemModelVersion(Base):
    __tablename__ = "system_model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_system_model_version_name"),
        UniqueConstraint("model_id", "id", name="uq_system_model_version_owner"),
        CheckConstraint(
            "status IN ('UPLOADING', 'BUILDING', 'VALIDATING', 'AVAILABLE', 'BUILD_FAILED', 'VALIDATION_FAILED', 'OFFLINE')",
            name="ck_system_model_version_status",
        ),
        CheckConstraint("status <> 'AVAILABLE' OR image_digest IS NOT NULL", name="ck_system_model_version_available_identity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("system_models.id", ondelete="RESTRICT"))
    version: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(32))
    framework: Mapped[str] = mapped_column(String(100))
    input_description: Mapped[str] = mapped_column(Text)
    input_bands: Mapped[list[int]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    resource_requirements: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    code_config_sha256: Mapped[str] = mapped_column(String(64))
    weight_sha256: Mapped[str] = mapped_column(String(64))
    image_digest: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
