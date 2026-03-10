from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, Date, Text, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime
from Base import Base

class DetectionTask(Base):
    __tablename__ = "detection_tasks"

    id              = Column(Integer, primary_key=True, index=True)
    task_name       = Column(String(255))
    before_image_id = Column(Integer, ForeignKey("images.id"))
    after_image_id  = Column(Integer, ForeignKey("images.id"))
    task_type       = Column(String(50))   # 影像比对 / 地物识别 / 变化检测
    status          = Column(String(20))   # 待处理 / 处理中 / 已完成 / 失败
    result_path     = Column(Text)
    created_time    = Column(DateTime, default=datetime.now)
    finished_time   = Column(DateTime)

    before_image     = relationship("Image", foreign_keys=[before_image_id], back_populates="before_tasks")
    after_image      = relationship("Image", foreign_keys=[after_image_id],  back_populates="after_tasks")
    analysis_results = relationship("AnalysisResult", back_populates="task")