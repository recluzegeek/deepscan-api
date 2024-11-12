from sqlalchemy import Float, Column, ForeignKey, String, DateTime, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Video(Base):
    __tablename__ = 'videos'

    id = Column(UUID(as_uuid=True), primary_key=True)
    filename = Column(String)
    video_path = Column(String)
    video_status = Column(String)
    predicted_class = Column(String, nullable=True)
    prediction_probability = Column(Float, nullable=True)