import os
import torch
from datetime import datetime
from torchvision import transforms
from .video_processing import VideoProcessor

transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Classification:
    def __init__(self, model, frames_path):
        self.model = model
        self.frames_path = frames_path
        self.video_processor = VideoProcessor(self.frames_path)
        self.face_images_with_original_frames = self.video_processor.extract_faces()

        print(f'{datetime.now()} - Creating Classification Instance for {os.path.basename(self.frames_path)}')

    def preprocess(self, image):
        print(f'{datetime.now()} - Applying Transformation on {os.path.basename(self.frames_path)}')
        return transformations(image)

    def infer(self):
        self.model.eval()
        results = []
        print(f'{datetime.now()} - Running Inference on {os.path.basename(self.frames_path)}')
        with torch.no_grad():
            for face_img, original_frame in self.face_images_with_original_frames:
                input_tensor = self.preprocess(face_img)
                input_tensor = input_tensor.unsqueeze(0)

                # Perform inference
                output = self.model(input_tensor)
                results.append((output, input_tensor, original_frame))

        return results
