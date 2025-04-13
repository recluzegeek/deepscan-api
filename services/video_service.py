import os
import json
import torch
import requests
from typing import Tuple
from datetime import datetime
from .frame_services import Frames
from ..utils.model_manager import ModelManager
from ..utils.classification import Classification

class VideoService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.frames = Frames()
        self.frames_abspath = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                self.config['paths']['frames_path'], 
                frames_path
            )
        )
        self.config = self.model_manager._load_config()
    
    def process_video(self, frames_path: str) -> Tuple[str, float]:
        # Download frames from MinIO
        self.frames.download_frames(frames_path)
        # Process frames
        # full_frames_path = os.path.abspath(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         self.config['paths']['frames_path'], 
        #         frames_path
        #     )
        # )

        print(f'{datetime.now()} - Processing frames at: {self.frames_abspath}')
        
        classifier = Classification(
            self.model_manager.model, 
            frames_path=self.frames_abspath, 
            cam=self.model_manager.cam
        )
        
        inference_results = classifier.infer()
        return self._process_results(inference_results)
    
    def _process_results(self, inference_results) -> Tuple[str, float]:
        all_probabilities = []
        for result in inference_results:
            probabilities = torch.softmax(result, dim=1)
            all_probabilities.append(probabilities)

        all_probabilities = torch.cat(all_probabilities, dim=0)
        mean_probabilities = all_probabilities.mean(dim=0)

        final_classification_index = torch.argmax(mean_probabilities).item()
        final_classification = "real" if final_classification_index == 0 else "fake"
        
        return final_classification, mean_probabilities[final_classification_index].item()
    
    def notify_completion(self, video_id: str, classification: str, probability: str):
        url = f"{self.config['laravel']['base_url']}{self.config['laravel']['inference_endpoint'].format(video_id=video_id)}"
        print('Sending results to ', url)

        # Prepare the JSON payload
        payload = {
            'classification': classification,
            'probability': probability
        }

        print('json payload: ', payload)
        
        # Send the request with JSON payload
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            print(response)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None