# TODO - Sign the request issued by FastAPI to Laravel Endpoint

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

from ..utils.database import get_db
from ..utils.models import Video
from ..services.video_service import VideoService

router = APIRouter()
video_service = VideoService()

class VideoModel(BaseModel):
    frames_path: str
    video_id: str

@router.post("/upload")
async def upload_video(data: VideoModel, db: Session = Depends(get_db)):
    try:
        video_id = UUID(data.video_id)
        update_video_status(video_id=video_id, video_status='processing', db=db)
        
        classification, probability = video_service.process_video(data.frames_path)
        
        update_video_status(video_id, 'completed', db)
        update_video_results(video_id, classification, probability, db)
        
        response = video_service.notify_completion(str(video_id))
        print(response.json(), response.status_code)

        return {
            "details": {
                "path": data.frames_path,
                "classification": classification,
                "probability": probability
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def update_video_status(video_id: str, video_status: UUID, db: Session):
    get_video(video_id, db).video_status = video_status
    db.commit()

def update_video_results(video_id: str, result: str, probability: float, db: Session):
    video = get_video(video_id, db)
    video.predicted_class = result
    video.prediction_probability = probability
    db.commit()

def get_video(video_id: str, db: Session):
    return db.query(Video).filter(Video.id == video_id).first()
