import torch
import os
from torchvision import transforms
from .video_processing import VideoProcessor
from datetime import datetime

transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Classification:
    def __init__(self, model, frames_path):
        self.model = model
        self.frames_path = frames_path
        self.face_images = VideoProcessor(self.frames_path).extract_faces()

        print(f'{datetime.now()} - Creating Classification Instance for {os.path.basename(self.frames_path)}')

    def preprocess(self, idx, total, image):
        print(f'{datetime.now()} - Applying Transformation on {os.path.basename(self.frames_path)} - {idx + 1} / {total}')
        return transformations(image)
    
    def infer(self):
        self.model.eval()
        results = []
        print(f'{datetime.now()} - Running Inference on {os.path.basename(self.frames_path)}')
        with torch.no_grad():
            for idx, face in enumerate(self.face_images):

                input_tensor = self.preprocess(idx, len(self.face_images), face)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                # Perform inference
                output = self.model(input_tensor)
                results.append(output)

        return results
