from sqlalchemy import Column, Integer, String, Date, Text, DateTime, ForeignKey,Numeric, BigInteger
from sqlalchemy.orm import relationship, DeclarativeBase
from geoalchemy2 import Geometry
from datetime import datetime
from models.Base import Base

class Image(Base):
    __tablename__ = "images"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(BigInteger, ForeignKey("user_info.id", ondelete="CASCADE"), nullable=False)
    image_name    = Column(String(255), nullable=False)
    resolution    = Column(Numeric(10, 4))
    capture_date  = Column(Date)
    satellite     = Column(String(200))
    image_type    = Column(String(50))
    region_code   = Column(String(20))
    boundary      = Column(Geometry(geometry_type='POLYGON', srid=4326))
    img_path      = Column(Text)
    upload_time   = Column(DateTime, default=datetime.now)

    # 关联 boundary_files
    boundary_files = relationship("BoundaryFile", back_populates="image")
    # before_tasks   = relationship("DetectionTask", foreign_keys="DetectionTask.before_image_id", back_populates="before_image")  # 待后续实现
    # after_tasks    = relationship("DetectionTask", foreign_keys="DetectionTask.after_image_id",  back_populates="after_image")   # 待后续实现


class BoundaryFile(Base):
    __tablename__ = "boundary_files"

    id           = Column(Integer, primary_key=True, index=True)
    image_id     = Column(Integer, ForeignKey("images.id"), nullable=False)
    file_prefix  = Column(String(255))
    shp_path     = Column(Text)
    dbf_path     = Column(Text)
    prj_path     = Column(Text)

    # 关联 images
    image = relationship("Image", back_populates="boundary_files")