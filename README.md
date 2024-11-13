# DeepScan API

DeepScan API is a FastAPI-based service that processes videos to detect deepfake content using deep learning models. It provides real-time analysis and visualization of the detection process.

## Features

- Deep learning-based deepfake detection
- GradCAM visualization for model interpretability
- Efficient frame processing and storage
- REST API endpoints for video processing
- Integration with Laravel backend
- Configurable settings via YAML

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Storage space for processed frames

## Installation

1. Clone the repository:

```bash
git clone https://github.com/recluzegeek/deepscan-api.git

cd deepscan-api
```

2. Create and activate a virtual environment:

```bash
python -m venv deepscan-api-venv
source deepscan-api-venv/bin/activate
```

OR

```cmd
.\deepscan-api-venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure settings:

```bash
cp config/settings.yaml.example config/settings.yaml
```

Edit the settings.yaml file to configure the API host, port, and other parameters.

## Project Structure

```text
deepscan-api/
├── config/
│ ├── settings.example.yaml
│ └── settings.yaml
├── utils/
│ ├── model_manager.py
│ ├── database.py
│ ├── models.py
│ └── classification.py
├── services/
│ └── video_service.py
├── routers/
│ └── video_router.py
├── models/
│ └── weights/
├── requirements.txt
└── README.md
```

## Usage

1. Run the API:

```python
fastapi run main.py --reload --host 0.0.0.0 --port 8000
```

2. API Endpoints:

- POST `/upload`: Process video frames for deepfake detection

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the MIT License.
