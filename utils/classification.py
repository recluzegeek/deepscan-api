import torch
import os
from torchvision import transforms
from .video_processing import VideoProcessor
from datetime import datetime

# Define the transformations
transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Classification:
    def __init__(self, model, video_path):
        self.model = model
        self.video_path = video_path
        self.face_images = VideoProcessor(self.video_path).extract_frames_and_faces()
        
        print(f'{datetime.now()} - Creating Classification Instance for {os.path.basename(self.video_path)}')

    def preprocess(self, image):
        print(f'{datetime.now()} - Applying Transformations on {os.path.basename(self.video_path)}')
        return transformations(image)
    
    def infer(self):
        self.model.eval()
        results = []
        print(f'{datetime.now()} - Running Inference on {os.path.basename(self.video_path)}')
        with torch.no_grad():
            for face in self.face_images:
                # Preprocess the image
                input_tensor = self.preprocess(face)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                # Perform inference
                output = self.model(input_tensor)
                results.append(output)

        return results
