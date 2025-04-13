# DeepScan API

DeepScan API is a FastAPI-based service that processes videos to detect deepfake content using deep learning models. It provides real-time analysis and visualization of the detection process.

## Features

- Deep learning-based deepfake detection
- GradCAM visualization for model interpretability
- Efficient frame processing and MinIO storage
- Integration with Laravel backend
- Configurable settings via YAML

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

## Vagrant Box

Vagrantfile for this repository can be found out at [recluzegeek/deepscan-web](https://github.com/recluzegeek/deepscan-web), under `vagrant/fastapi.sh`. The script is used for provisioning the ubuntu jammy configured to host this API, in a multivm vagrant environment, but it can be configured according to your needs.

## Local Installation

1. Clone the repository:

```bash
git clone https://github.com/recluzegeek/deepscan-api.git
cd deepscan-api
```

1. Create and activate a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
```

OR for windows

```cmd
python -m venv .venv && .\venv\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure settings:

```bash
`cp config/settings.example.yaml config/settings.yaml
````

Edit the settings.yaml file to configure the API host, port, and other parameters.

## Project Structure

```text
.
├── config
│   ├── settings.example.yaml
│   └── settings.yaml
├── models
│   ├── __init__.py
│   └── swin_model.pth
├── routers
│   ├── __init.py__
│   └── video_router.py
├── services
│   ├── frame_services.py
│   └── video_service.py
├── temp-frames
├── utils
│   ├── classification.py
│   ├── __init.py__
│   ├── model_manager.py
│   └── video_processing.py
├── Dockerfile
├── __init__.py
├── main.py
├── README.md
└── requirements.txt
```

## Usage

1. Run the API:

```python
fastapi run main.py --reload --host 0.0.0.0 --port 9010
```

1. API Endpoints:

- POST `/upload`: Process video frames for deepfake detection

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
