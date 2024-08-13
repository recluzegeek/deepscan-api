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

from pytorch_grad_cam import GradCAM

from ..utils.database import get_db
from ..utils.models import VideoClassification
from ..utils.classification import Classification

router = APIRouter()

class Video(BaseModel):
    frames_path: str
    video_id: str

@router.post("/upload")
async def upload_video(data: Video, db: Session = Depends(get_db)):
    try:
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
        
        save_results(video_id=data.video_id, predicted_class=final_classification, prediction_probability=mean_probabilities[final_classification_index].item(), db=db)

        return {
            "details": {
                "path": data.frames_path,
                "classification": final_classification,
                "probability": mean_probabilities[final_classification_index].item()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def save_results(video_id: str, predicted_class: str, prediction_probability: float, db: Session):
    try:

        results = VideoClassification(
            id=uuid.uuid4(),
            video_id=uuid.UUID(video_id),
            predicted_class=predicted_class,
            prediction_probability=prediction_probability
        )

        db.add(results)
        db.commit()
        db.refresh(results)

        return results
    
    except Exception as e:

        print(f"Error saving results: {str(e)}") 
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save results.")

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
