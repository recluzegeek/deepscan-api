from sqlalchemy import Float, Column, ForeignKey, Integer, String, DateTime, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Video(Base):
    __tablename__ = "uploaded_videos"

    id = Column(UUID, primary_key=True)
    filename = Column(String(255))
    video_storage_path = Column(String(255))
    status = Column(String(255))
    user_id = Column(UUID)
    upload_date_time = Column(DateTime(timezone=True), default=func.now())
    
    video_result = relationship('Classification', back_populates="video_source")
    video_report = relationship('AnalysisReport', back_populates="video_source")

class Classification(Base):
    __tablename__ = "video_results"

    id = Column(UUID, primary_key=True)
    predicted_class = Column(Integer)
    prediction_probability = Column(Float)
    video_id = Column(UUID, ForeignKey('uploaded_videos.id'))
    
    video_source = relationship('Video', back_populates="video_result")
    video_report = relationship('AnalysisReport', back_populates="classification_source")

class AnalysisReport(Base):
    __tablename__ = "video_result_reports"

    id = Column(UUID, primary_key=True)
    report_url = Column(String(255))
    classification_id = Column(UUID, ForeignKey('video_results.id'))
    video_id = Column(UUID, ForeignKey('uploaded_videos.id'))

    video_source = relationship('Video', back_populates="video_report")
    classification_source = relationship('Classification', back_populates="video_report")
