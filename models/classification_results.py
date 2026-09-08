from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)

from models.Base import Base


class ClassificationResult(Base):
    __tablename__ = "classification_results"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "identity_sha256",
            name="uq_classification_results_user_identity",
        ),
        CheckConstraint(
            "status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')",
            name="ck_classification_results_status",
        ),
        CheckConstraint(
            "area_status IN ('NOT_COMPUTED', 'SUCCEEDED', 'FAILED')",
            name="ck_classification_results_area_status",
        ),
        Index("ix_classification_results_user_status", "user_id", "status"),
    )

    id = Column(String(32), primary_key=True)
    user_id = Column(
        BigInteger,
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_image_id = Column(
        Integer,
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_model_id = Column(
        Integer,
        ForeignKey("model_library.id", ondelete="SET NULL"),
        nullable=True,
    )
    identity_sha256 = Column(String(64), nullable=False)
    image_content_sha256 = Column(String(64), nullable=False)
    weight_content_sha256 = Column(String(64), nullable=False)
    inference_parameters = Column(JSON, nullable=False)
    classification_scheme_version = Column(String(64), nullable=False)
    pipeline_version = Column(String(64), nullable=False)
    grid_policy_version = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    lease_owner = Column(String(128), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    failure_detail = Column(Text, nullable=True)
    classes_path = Column(Text, nullable=True)
    valid_mask_path = Column(Text, nullable=True)
    crs = Column(Text, nullable=True)
    transform = Column(JSON, nullable=True)
    raster_width = Column(Integer, nullable=True)
    raster_height = Column(Integer, nullable=True)
    resolution = Column(JSON, nullable=True)
    bounds = Column(JSON, nullable=True)
    generation_metrics = Column(JSON, nullable=True)
    class_area_m2 = Column(JSON, nullable=True)
    area_status = Column(
        String(16),
        nullable=False,
        default="NOT_COMPUTED",
        server_default="NOT_COMPUTED",
    )
    area_completed_at = Column(DateTime(timezone=True), nullable=True)
    area_failure_detail = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
