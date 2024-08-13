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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, XGradCAM

from ..utils.database import get_db
from ..utils.models import VideoClassification
from ..utils.classification import Classification
from PIL import Image

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
            # cv2.rectangle(original_frame, (y,y+h), (x,x+w), (0, 255, 0), 2)
            cv2.imwrite('original_frame.jpg', original_frame)
            # return
            image = cv2.rectangle(original_frame, (x, y), (x + w, y + h), (36,255,12), 1)
            captioned_image = cv2.putText(image, 'Detected Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # cv2.imshow('captioned_image.jpg', captioned_image)
            cv2.imwrite('captioned_image.jpg', captioned_image)

            return




            # rgb_img = cv2.imread(face_img, 1)[:, :, ::-1]
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

            print(cam.outputs)
            all_probabilities.append(cam.outputs)

            # Save the result
            save_path = f'/mnt/win/deepscan-api/visualized_frames/{data.video_id}_{idx}.jpg'
            cv2.imwrite(save_path, cam_image)


            # final_classification_index = torch.argmax(probabilities).item()
            # targets = [ClassifierOutputTarget(final_classification_index)]

            # Generate the GradCAM
            # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # # https://github.com/jacobgil/pytorch-grad-cam/issues/365
            # #input_tensor has shape (1, 3, 224, 224)
            # input_tensor = np.array(input_tensor)
            # input_tensor_float = input_tensor.squeeze(0).transpose(1, 2, 0).astype(np.float32) / 255.0

            # print("Input Tensor Shape after processing:", input_tensor_float.shape)  # Should print (224, 224, 3)

            # # Assuming grayscale_cam is in the shape of (1, 224, 224)
            # grayscale_cam = grayscale_cam.squeeze(0)  # Now shape is (224, 224)
            # print("Grayscale CAM Shape after squeeze:", grayscale_cam.shape)

            # # Ensure values are between 0 and 1
            # grayscale_cam = np.clip(grayscale_cam, 0, 1)

            # # Repeat grayscale_cam to match the input image shape
            # grayscale_cam = np.repeat(grayscale_cam[..., np.newaxis], 3, axis=-1)  # Shape becomes (224, 224, 3)
            # print("Grayscale CAM Shape after repeat:", grayscale_cam.shape)

            # visualization = show_cam_on_image(input_tensor_float, grayscale_cam)

            # print('after visualization')



                # Ensure the original frame is in the correct format
                # original_frame_float = original_frame.astype(np.float32) / 255.0

                # Get the face coordinates
                # x, y, w, h = classifier.video_processor.face_coordinates[idx]

                # Resize the GradCAM output to match the face region size
                # grayscale_cam_resized = cv2.resize(grayscale_cam[0], (w, h))  # Resize to the width and height of the face region

            # # Normalize the GradCAM output to [0, 1]
            # grayscale_cam_resized = np.clip(grayscale_cam_resized, 0, 1)  # Ensure values are in [0, 1]

            # # Convert the normalized GradCAM output to a format suitable for visualization
            # grayscale_cam_resized = (grayscale_cam_resized - np.min(grayscale_cam_resized)) / (np.max(grayscale_cam_resized) - np.min(grayscale_cam_resized) + 1e-10)  # Normalize to [0, 1]

            # # Overlay the GradCAM visualization back onto the original frame
            # overlay = show_cam_on_image(original_frame_float[y:y+h, x:x+w], grayscale_cam_resized)  # Overlay on the cropped area

            # # Check the overlay output
            # print(f"Overlay shape: {overlay.shape}")
            # print(f"Overlay values: {overlay}")

            # # Place the overlay back into the original frame
            # original_frame_float[y:y+h, x:x+w] = overlay

            # # Save the GradCAM-visualized frame
            # save_path = f'/mnt/win/deepscan-api/visualized_frames/{data.video_id}_{idx}.jpg'
            # cv2.imwrite(save_path, overlay)
            # cv2.imwrite(save_path, original_frame_float * 255)  # Convert back to [0, 255] range for saving


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
