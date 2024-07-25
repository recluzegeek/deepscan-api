# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

import torch
from ..utils.classification import Classification
import timm

from datetime import datetime

from fastapi import APIRouter, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from ..utils.models import Video
from ..utils.database import get_db
import os
import shutil
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_video(user_id: str, uploaded_video: UploadFile, db: Session = Depends(get_db)):
    try:
        file_path = await save_video(user_id, uploaded_video, db)
        print(f'{datetime.now()} - Received file: {os.path.basename(file_path)}')
        response_message = f"{uploaded_video.filename} video has been saved at {file_path}"
        
        print(f'{datetime.now()} - {os.path.basename(file_path)} has been saved at {file_path}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'{datetime.now()} - Loading Model on {os.path.basename(file_path)}')

        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
        model.load_state_dict(torch.load('/mnt/win/deepscan-api/models/10_epochs.pth', map_location=device))

        print(f'{datetime.now()} - Model weights loaded')

        model.eval()

        classifier = Classification(model, video_path=file_path)
        print(f'{datetime.now()} - Called Inference on {os.path.basename(file_path)}')
        inference_results = classifier.infer()

        print(f'{datetime.now()} - Processing Results of {os.path.basename(file_path)}')
        # Process the results as needed
        all_probabilities = []
        for result in inference_results:
            probabilities = torch.softmax(result, dim=1)
            all_probabilities.append(probabilities)
            print(probabilities)

        all_probabilities = torch.cat(all_probabilities, dim=0)
        mean_probabilities = all_probabilities.mean(dim=0)

        final_classification_index = torch.argmax(mean_probabilities).item()
        final_classification = "real" if final_classification_index == 0 else "fake"

        print(f"Final classification for the video: {final_classification}")
        print(f"Mean probabilities: {mean_probabilities}")


        return {
            "details": {
                "message": response_message,
                "path": file_path,
                "classification": final_classification,
                "probability": mean_probabilities[0 if final_classification == "real" else 1]
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
        user_id_uuid = uuid.UUID(user_id)

        new_video = Video(
            id=uuid.uuid4(),
            user_id=user_id_uuid,
            filename=os.path.basename(file_path),
            video_storage_path=file_path,
            status='new'
        )
        
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        return new_video
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid user ID format. {e}")