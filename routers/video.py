# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

import torch
from ..utils.classification import Classification
import timm

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

router = APIRouter()

class Data(BaseModel):
    frames_path: str

@router.post("/upload")
async def upload_video(data: Data):
    try:
        frames_path = f'/mnt/win/deepscan-web/storage/app/frames/{data.frames_path}'
        print(f'{datetime.now()} - Frame path for {data.frames_path} received - {frames_path}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'{datetime.now()} - Loading Model on {os.path.basename(frames_path)}')

        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
        model.load_state_dict(torch.load('/mnt/win/deepscan-api/models/10_epochs.pth', map_location=device))

        print(f'{datetime.now()} - Model weights loaded')
        torch.manual_seed(42)
        model.eval()

        classifier = Classification(model, frames_path=frames_path)
        print(f'{datetime.now()} - Called Inference on {os.path.basename(frames_path)}')
        inference_results = classifier.infer()

        print(f'{datetime.now()} - Processing Results of {os.path.basename(frames_path)}')

        all_probabilities = []
        for result in inference_results:
            probabilities = torch.softmax(result, dim=1)
            all_probabilities.append(probabilities)
            print(probabilities)
        
        # print(f'all_probabilities: {all_probabilities}')
        # print(f'shape is: {all_probabilities[0].shape}')

        all_probabilities = torch.cat(all_probabilities, dim=0)
        mean_probabilities = all_probabilities.mean(dim=0)

        final_classification_index = torch.argmax(mean_probabilities).item()
        final_classification = "real" if final_classification_index == 0 else "fake"

        print(f"Final classification for {os.path.basename(data.frames_path)}: {final_classification}")
        print(f"Mean probabilities: {mean_probabilities}")


        return {
            "details": {
                "path": data.frames_path,
                "classification": final_classification,
                "probability": mean_probabilities[final_classification_index].item()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
