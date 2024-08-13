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

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

from ..utils.database import get_db
from ..utils.models import VideoClassification
from ..utils.classification import Classification

router = APIRouter()

class Video(BaseModel):
    frames_path: str
    video_id: str

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



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
        # torch.manual_seed(42)
        model.eval()

        classifier = Classification(model, frames_path=frames_path)
        print(f'{datetime.now()} - Called Inference on {os.path.basename(frames_path)}')
        # inference_results = classifier.infer()

        print(f'{datetime.now()} - Processing Results of {os.path.basename(frames_path)}')

        all_probabilities = []

        for idx, (face_img, original_frame) in enumerate(classifier.face_images_with_original_frames):

            # check whether the face_img is an PIL Image or Numpy Array
            print(type(face_img))
            print(type(original_frame))
            print(original_frame.shape)

            # Get face coords
            x, y, w, h = classifier.video_processor.face_coordinates[idx]
            print(x, y, w, h)

            # Draw border around the original_frame using x, y, w, h
            cv2.imwrite('original_frame.jpg', original_frame)
            cv2.imwrite('cropped_face.jpg', face_img)

            image = cv2.rectangle(original_frame, (x, y), (x + w, y + h), (36,255,12), 1)
            captioned_image = cv2.putText(image, 'Localized Domain', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite('detected_face_in_original_frame.jpg', captioned_image)

            # Get the height, width of the cropped face image, so we've to 
            # resize the gradcam visualized image back to the original shape
            # before overlaying onto the original frame

            height, width, channels = face_img.shape
            print(f'Original Cropped Face Dimensions...{face_img.shape}')

            rgb_img = cv2.resize(face_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=True,
                        aug_smooth=True)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cam_image_resized = cv2.resize(cam_image, (width, height))

            cv2.imwrite('cam_image_original.jpg', cam_image)
            cv2.imwrite('cam_image_resized.jpg', cam_image_resized)

            # Overlay the cam_image_resized on the original_frame
            # on the face coords of the original_frame

            # Select the original_frame region outside of the face
            # so, we can overlay

            # Create a mask for overlaying
            shapes = np.zeros_like(original_frame, np.uint8)

            # Overlay the resized CAM image onto the original frame
            shapes[y:y+h, x:x+w] = cam_image_resized

            alpha = 0.4
            mask = shapes.astype(np.uint8)

            # Overlay the cam_image_resized on the original_frame
            original_frame = cv2.addWeighted(original_frame, alpha, mask, 1 - alpha, 0)

            # Save the final visualized frame
            cv2.imwrite('visualized_frame.jpg', original_frame)
            # overlay = cv2.addWeighted(original_frame[y:y+h, x:x+w], 0.5, cam_image_resized, 0.5,0 )
            # original_frame[y:y+h, x:x+w] = overlay

            print(cam.outputs)
            all_probabilities.append(cam.outputs)

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
