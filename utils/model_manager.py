import os
import timm
import requests
import tqdm
import torch
from pytorch_grad_cam import GradCAM
from typing import List

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config = self._load_config()

        # 🔧 Build absolute path: from utils → ../.. → root → models
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        if not self.config['model']['weights_download_path']:
            self.config['model']['weights_download_path'] = os.path.join(
                models_dir, "swin_model.pth"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._download_weights()
        self.model = self._load_model()
        self.cam = self._initialize_grad_cam()
    
    def _load_config(self):
        # Load config from YAML file
        import yaml
        with open("config/settings.yaml") as f:
            return yaml.safe_load(f)

    def _download_weights(self):
        remote_url = self.config['model']['weights_remote_path']
        local_path = self.config['model']['weights_download_path']

        if not os.path.exists(local_path):
            print(f"⬇️ Downloading model weights from {remote_url}")

            response = requests.get(remote_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KiB

            with open(local_path, 'wb') as f, tqdm.tqdm(
                desc="📦 Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))

            print(f"✅ Weights downloaded and saved to {local_path}")
        else:
            print(f"📁 Weights already exist at {local_path}")

    def _load_model(self):
        model = timm.create_model(
            self.config['model']['name'], 
            pretrained=False, 
            num_classes=self.config['model']['num_classes']
        )

        model.load_state_dict(
            torch.load(self.config['model']['weights_download_path'], 
            map_location=self.device)
        )
        return model
    
    def _initialize_grad_cam(self):
        target_layers = [self.model.layers[-1].blocks[-1].norm2]
        return GradCAM(
            model=self.model, 
            target_layers=target_layers, 
            reshape_transform=self.reshape_transform
        )
    
    @staticmethod
    def reshape_transform(tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result