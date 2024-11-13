import timm
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.cam = self._initialize_grad_cam()
    
    def _load_config(self):
        # Load config from YAML file
        import yaml
        with open("config/settings.yaml") as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        model = timm.create_model(
            self.config['model']['name'], 
            pretrained=True, 
            num_classes=self.config['model']['num_classes']
        )
        model.load_state_dict(
            torch.load(self.config['model']['weights_path'], 
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