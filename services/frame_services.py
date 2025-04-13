import os
from minio import Minio
from minio.error import S3Error

class MinioClient:
    def __init__(self):
        self.config = self._load_config()
        self.client = Minio(self.config['minio']['endpoint'],
            access_key=self.config['minio']['access_key'],
            secret_key=self.config['minio']['secret_key'],
            secure=self.config['minio']['secure'],
            region=self.config['minio']['region'],
        )
        self.frames_bucket = self.config['minio']['frames_bucket']
        self.gradcam_frames_bucket = self.config['minio']['gradcam_frames_bucket']

    def _load_config(self):
        # Load config from YAML file
        import yaml
        with open("config/settings.yaml") as f:
            return yaml.safe_load(f)


    def list_objects(self, prefix, bucket_name = None):
        try:
            if bucket_name is None:
                bucket_name = self.frames_bucket
            return self.client.list_objects(bucket_name, prefix=prefix)
        except S3Error as err:
            print(f"Error listing objects: {err}")
            return []

    def fget_object(self, object_name, file_path, bucket_name = None):
        try:
            if bucket_name is None:
                bucket_name = self.frames_bucket
            self.client.fget_object(bucket_name, object_name, file_path)
        except S3Error as err:
            print(f"Error downloading object: {err}")

class Frames:
    def __init__(self):
        self.client = MinioClient()

    def download_frames(self, frames_pattern):
        download_path = os.path.join(os.path.dirname(__file__), "../temp-frames")
        os.makedirs(download_path, exist_ok=True)
        objects = self.client.list_objects(prefix=frames_pattern)
        for obj in objects:
            self.client.fget_object(obj.object_name, os.path.join(download_path, obj.object_name))
        print(f"{datetime.now()} - Downloaded frames matching pattern: {frames_pattern}")
