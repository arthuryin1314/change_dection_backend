from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, Text

from models.Base import Base


class MLModel(Base):
    __tablename__ = "model_library"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user_info.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=False)
    weight_file_path = Column(Text, nullable=False)
    model_file_path = Column(Text, nullable=False)
    description = Column(Text)
    weight_content_sha256 = Column(String(64), nullable=True)
    weight_content_sha256_size = Column(BigInteger, nullable=True)
    weight_content_sha256_mtime_ns = Column(BigInteger, nullable=True)
    upload_time = Column(DateTime, default=datetime.now)
    updated_time = Column(DateTime, default=datetime.now, onupdate=datetime.now)
