# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

from fastapi import APIRouter, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from ..utils.models import Video
from ..utils.database import get_db
from datetime import datetime
import os
import shutil
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_video(user_id: str, uploaded_video: UploadFile, db: Session = Depends(get_db)):
    try:
        file_path = await save_video(user_id, uploaded_video, db)
        response_message = f"{uploaded_video.filename} video has been saved at {file_path}"
        return {
            "details": {
                "message": response_message,
                "path": file_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def save_video(user_id: str, video_file: UploadFile, db: Session):
    upload_directory = '/mnt/win/deepscan-api/storage/uploaded_videos'
    os.makedirs(upload_directory, exist_ok=True)
    file_path = os.path.join(upload_directory, video_file.filename)

    # Save the uploaded video file
    with open(file_path, 'wb') as video_dst:
        shutil.copyfileobj(video_file.file, video_dst)

    # Save video metadata in the database
    save_video_in_db(user_id=user_id, file_path=file_path, db=db)

    return file_path

def save_video_in_db(user_id: str, file_path: str, db: Session):
    try:
        # Convert user_id to UUID
        user_id_uuid = uuid.UUID(user_id)  # Convert the user_id string to a UUID

        # Generate a new UUID for the video
        new_video = Video(
            id=uuid.uuid4(),  # Generate a new UUID
            user_id=user_id_uuid,  # Use the UUID here
            filename=os.path.basename(file_path),  # Use actual file name
            video_storage_path=file_path,
            status='new'
        )
        
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        return new_video
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid user ID format.")