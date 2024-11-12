# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

import os
import uuid
import timm
import torch
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, Depends

from uuid import UUID
from pytorch_grad_cam import GradCAM

from ..utils.database import get_db
from ..utils.models import Video
from ..utils.classification import Classification

router = APIRouter()

class VideoModel(BaseModel):
    frames_path: str
    video_id: str

@router.post("/upload")
async def upload_video(data: VideoModel, db: Session = Depends(get_db)):
    try:
        video_id = UUID(data.video_id)  # change back to uuid from string
        update_video_status(video_id = video_id, video_status = 'processing', db = db)
        frames_path = f'/mnt/win/deepscan-web/storage/app/frames/{data.frames_path}'
        print(f'\n\n{datetime.now()} - Frame path for {data.frames_path} received - {frames_path}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'{datetime.now()} - Loading Model on {os.path.basename(frames_path)}')

        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
        model.load_state_dict(torch.load('/mnt/win/deepscan-api/models/10_epochs.pth', map_location=device))

        target_layers = [model.layers[-1].blocks[-1].norm2]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

        print(f'{datetime.now()} - Model weights loaded')

        classifier = Classification(model, frames_path=frames_path, cam=cam)
        print(f'{datetime.now()} - Called Inference on {os.path.basename(frames_path)}')
        inference_results = classifier.infer()

        print(f'{datetime.now()} - Processing Results of {os.path.basename(frames_path)}')

        all_probabilities = []
        for result in inference_results:
            probabilities = torch.softmax(result, dim=1)
            all_probabilities.append(probabilities)
            # print(probabilities)

        all_probabilities = torch.cat(all_probabilities, dim=0)
        mean_probabilities = all_probabilities.mean(dim=0)

        final_classification_index = torch.argmax(mean_probabilities).item()
        final_classification = "real" if final_classification_index == 0 else "fake"

        print(f"Final classification for {os.path.basename(data.frames_path)}: {final_classification}")
        print(f"Mean probabilities: {mean_probabilities}\n\n")
        
        update_video_status(video_id, 'completed', db)
        update_video_results(video_id, final_classification, mean_probabilities[final_classification_index].item(), db)

        return {
            "details": {
                "path": data.frames_path,
                "classification": final_classification,
                "probability": mean_probabilities[final_classification_index].item()
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

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result
