# SQLAlchemy models handle database interactions
# SQLAlchemy models to manage database operations


from sqlalchemy import Float, Column, ForeignKey, Integer, String, DateTime, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Video(Base):
    __tablename__ = "uploaded_videos"

    id = Column(UUID, primary_key=True)
    file_name = Column(String)
    video_url = Column(String)
    upload_date_time = Column(DateTime(timezone=True), default=func.now())
    
    video_result = relationship('Classification', back_populates="video_source")
    video_report = relationship('AnalysisReport', back_populates="video_source")


class Classification(Base):
    __tablename__ = "video_results"

    id = Column(UUID, primary_key=True)
    predicted_class = Column(Integer)
    prediction_probability = Column(Float)
    video_id = Column(String, ForeignKey('uploaded_videos.id'))
    
    video_source = relationship('Video', back_populates="video_result")
    video_report = relationship('AnalysisReport', back_populates="video_source")


class AnalysisReport(Base):
    __tablename__ = "video_result_reports"

    id = Column(UUID, primary_key=True)
    report_url = Column(String)
    classification_id = Column(String, ForeignKey('video_results.id'))

    video_source = relationship('Classification', back_populates="video_report")
