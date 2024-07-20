# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

from fastapi import APIRouter, UploadFile, HTTPException

import os
import shutil


router = APIRouter()

"""
    Video uploading endpoint. Takes an UploadFile() argument and stores in the local server storage.
    Saves the path where the video has been stored along with the unique video uuid().

    # TODO - Store video metadata in database
    # TODO - Make protected routes

"""
@router.post("/upload")
async def upload_video(uploaded_video: UploadFile = UploadFile(...)):

    try:
        upload_directory = '/mnt/win/deepscan-api/storage/uploaded_videos'
        os.makedirs(upload_directory, exist_ok=True)
        file_path = os.path.join(upload_directory, uploaded_video.filename)
        
        # Save the uploaded video file
        with open(file_path, 'wb') as video_dst:
            shutil.copyfileobj(uploaded_video.file, video_dst)
        
        response_message = f"{uploaded_video.filename} video has been saved at {file_path}"
        return {
            "details": {
                "message": response_message,
                "path": file_path
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


def save_video(video_name, stored_path):
    pass