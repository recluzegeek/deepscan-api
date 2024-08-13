import os
import cv2
import numpy as np
from datetime import datetime
from .video_processing import VideoProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class Classification:
    def __init__(self, model, frames_path, cam):
        self.model = model
        self.frames_path = frames_path
        self.cam = cam
        self.video_processor = VideoProcessor(self.frames_path)
        self.face_images_with_original_frames = self.video_processor.extract_faces()

        print(f'{datetime.now()} - Creating Classification Instance for {os.path.basename(self.frames_path)}')


    def infer(self):
        self.model.eval()
        results = []
        print(f'{datetime.now()} - Running Inference on {os.path.basename(self.frames_path)}')

        for idx, (face_img, original_frame) in enumerate(self.face_images_with_original_frames):

            # Get face coords
            x, y, w, h = self.video_processor.face_coordinates[idx]
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

            height, width = face_img.shape[:2]
            print(f'Original Cropped Face Dimensions...{face_img.shape}')
            rgb_img = cv2.resize(face_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
            grayscale_cam = self.cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=True,
                        aug_smooth=True)
            
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cam_image_resized = cv2.resize(cam_image, (width, height))
            cv2.imwrite('cam_image_original.jpg', cam_image)
            cv2.imwrite('cam_image_resized.jpg', cam_image_resized)

            # Overlay the cam_image_resized on the original_frame on the face coords of the original_frame
            # Select the original_frame region outside of the face so, we can overlay

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

            # print(self.cam.outputs)
            results.append(self.cam.outputs)

        return results
