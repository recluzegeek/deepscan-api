from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.video_service import VideoService

router = APIRouter()
video_service = VideoService()

class VideoModel(BaseModel):
    frames_path: str
    video_id: str

@router.post("/upload")
async def upload_video(data: VideoModel):
    try:

        classification, probability = video_service.process_video(data.frames_path)
        
        print('...classification done, now sending back to laravel endpoint...')
        response = video_service.notify_completion(data.video_id, classification, probability)
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
