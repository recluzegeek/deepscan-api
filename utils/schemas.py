# Pydantic schemas are about data validation and transfer
# Pydantic schemas to validate and serialize request and response data


from pydantic import BaseModel
from typing import Optional


class Video(BaseModel):
    """
    This model represents the uploaded video.
    Args:
        video_id: Randomly generated video id
        file_name: Extracted video file name
        user_id: Access token or user_id for association
        video_url: Path to the uploaded video on server / cloud
        upload_date_time: Moment video gets uploaded
    """
    video_id: str
    file_name: str
    user_id: str = 1
    video_url: str
    upload_date_time: str


class Classification(BaseModel):
    """
    Represents the Deepfake Detection
    Args:
        no_of_frames: Number of frames to be extracted from the video randomly
        predicted_class: Model Prediction
        prediction_probability: Probability of the model prediction
        report: Generated Report of the Video Analysis (includes classification results, manipulation visualization on cropped face images)
    """
    no_of_frames: int
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


