from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, Date, Text, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime
from Base import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id          = Column(Integer, primary_key=True, index=True)
    task_id     = Column(Integer, ForeignKey("detection_tasks.id"))
    region_code = Column(String(20))
    change_area = Column(Numeric(15, 4))
    change_rate = Column(Numeric(10, 4))
    change_type = Column(String(100))
    stat_time   = Column(DateTime, default=datetime.now)

    task = relationship("DetectionTask", back_populates="analysis_results")