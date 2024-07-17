from datetime import datetime
import json

class VideoClassification:
    video_id: str
    classification_criteria: json
    classification_results: json
    classification_date_time: datetime