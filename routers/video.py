# https://medium.com/cuddle-ai/async-architecture-with-fastapi-celery-and-rabbitmq-c7d029030377
# https://www.linode.com/docs/guides/task-queue-celery-rabbitmq/
# https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/

import os
import uuid
import timm
import torch
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, Depends

import cv2

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

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

        target_layers = [model.layers[-1].blocks[-1].norm1]

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers)

        print(f'{datetime.now()} - Model weights loaded')
        torch.manual_seed(42)
        model.eval()

        classifier = Classification(model, frames_path=frames_path)
        print(f'{datetime.now()} - Called Inference on {os.path.basename(frames_path)}')
        inference_results = classifier.infer()

        print(f'{datetime.now()} - Processing Results of {os.path.basename(frames_path)}')

        all_probabilities = []

        for idx, (result, input_tensor, original_frame) in enumerate(inference_results):
            probabilities = torch.softmax(result, dim=1)
            all_probabilities.append(probabilities)
            print(probabilities)

            # Apply GradCAM
            # final_classification_index = torch.argmax(probabilities).item()
            # targets = [ClassifierOutputTarget(final_classification_index)]
            
            # Generate the GradCAM
            grayscale_cam = cam(input_tensor=input_tensor)
            # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            
            # Ensure the original frame is in the correct format
            original_frame_float = original_frame.astype(np.float32) / 255.0

            # Resize the GradCAM output to match the original frame size
            grayscale_cam_resized = cv2.resize(grayscale_cam[0], (original_frame.shape[1], original_frame.shape[0]))

            # Visualize GradCAM on the original frame
            visualization = show_cam_on_image(original_frame_float, grayscale_cam_resized)  # Use the resized GradCAM image

            # Save the GradCAM-visualized frame
            save_path = f'/mnt/win/deepscan-api/visualized_frames/{data.video_id}_{idx}.jpg'
            cv2.imwrite(save_path, visualization)


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
