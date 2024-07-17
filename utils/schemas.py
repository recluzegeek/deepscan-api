from datetime import datetime
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional


class Video(BaseModel):
    """
    This model represents the uploaded video.
    Args:
        video_id: Randomly generated video id
        video_name: Extracted video file name
        user_id: User who uploaded the video
        upload_date_time: Moment video gets uploaded
    """
    video_id: str
    video_name: str
    user_id: str = 1
    upload_date_time: datetime = datetime.now().strftime("%X_%x_%f") # '12:24:03_07/17/24_523841'


class Classification(BaseModel):
    """
    Represents the Deepfake Detection
    Args:
        no_of_frames: Number of frames to be extracted from the video randomly
        feature_classes: 
        predicted_class: Model Prediction
        prediction_probability: Probability of the model prediction
        report: Generated Report of the Video Analysis (includes classification results, manipulation visualization on cropped face images)
    """
    no_of_frames: int
    feature_classes = Enum('real', 'fake', 'uncertain') # uncertain - no face detected
    predicted_class: int
    prediction_probability: float
    report: AnalysisReport


class AnalysisReport(BaseModel):
    """
    Report generated after video analysis
    Args:
        report_id: Randomly generated report id
    """
    report_id: str
    error: Optional[str]


