from sqlalchemy import Float, Column, ForeignKey, String, DateTime, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Video(Base):
    __tablename__ = "uploaded_videos"

    id = Column(UUID, primary_key=True, index=True)
    filename = Column(String(255))
    path = Column(String(255))
    status = Column(String(255))
    user_id = Column(UUID, ForeignKey('users.id'))
    upload_date_time = Column(DateTime(timezone=True), default=func.now())
    
    video_result = relationship('VideoClassification', back_populates="video_source")

class VideoClassification(Base):
    __tablename__ = "video_results"

    id = Column(UUID, primary_key=True)
    predicted_class = Column(String)
    prediction_probability = Column(Float)
    video_id = Column(UUID, ForeignKey('uploaded_videos.id'))
    
    video_source = relationship('Video', back_populates="video_result")
