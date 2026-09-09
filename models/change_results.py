from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)

from models.Base import Base


class ChangeResult(Base):
    __tablename__ = "change_results"
    __table_args__ = (
        UniqueConstraint("user_id", "request_id", name="uq_change_results_user_request"),
        CheckConstraint(
            "status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')",
            name="ck_change_results_status",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(36), nullable=False)
    user_id = Column(
        BigInteger,
        ForeignKey("user_info.id", ondelete="CASCADE"),
        nullable=False,
    )
    before_image_id = Column(Integer, ForeignKey("images.id", ondelete="SET NULL"))
    after_image_id = Column(Integer, ForeignKey("images.id", ondelete="SET NULL"))
    source_model_id = Column(Integer, ForeignKey("model_library.id", ondelete="SET NULL"))
    before_result_id = Column(String(32), ForeignKey("classification_results.id", ondelete="SET NULL"))
    after_result_id = Column(String(32), ForeignKey("classification_results.id", ondelete="SET NULL"))
    before_identity_sha256 = Column(String(64), nullable=False)
    after_identity_sha256 = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    lease_owner = Column(String(128), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))
    error_http_status = Column(Integer)
    error_code = Column(String(64))
    error_message = Column(Text)
    error_data = Column(JSON)
    before_snapshot = Column(JSON)
    after_snapshot = Column(JSON)
    matrix_m2 = Column(JSON)
    common_valid_area_m2 = Column(Float)
    crs = Column(Text)
    transform = Column(JSON)
    raster_width = Column(Integer)
    raster_height = Column(Integer)
    bounds = Column(JSON)
    before_window = Column(JSON)
    after_window = Column(JSON)
    calculated_at = Column(DateTime(timezone=True))
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
