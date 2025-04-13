import os
import io
from minio import Minio
from minio.error import S3Error
from datetime import datetime

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

    def get_frames(self, object_name, file_path, bucket_name = None):
        try:
            if bucket_name is None:
                bucket_name = self.frames_bucket
            self.client.fget_object(bucket_name, object_name, file_path)
        except S3Error as err:
            print(f"Error downloading object: {err}")

    def upload_image_stream_to_minio(self, bucket_name, object_name, data_stream, length, content_type="application/octet-stream"):
        try:
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data_stream,
                length=length,
                content_type=content_type
            )
            print(f"Uploaded {object_name} to {bucket_name}")
        except S3Error as err:
            print(f"Error uploading object {object_name}: {err}")


class Frames:
    def __init__(self):
        self.client = MinioClient()
        self.download_path = os.path.join(os.path.dirname(__file__), self.client.config['paths']['frames_path'])

    def download_frames(self, frames_pattern):
        os.makedirs(self.download_path, exist_ok=True)
        objects = self.client.list_objects(prefix=frames_pattern)
        for obj in objects:
            self.client.get_frames(obj.object_name, os.path.join(self.download_path, obj.object_name))
        print(f"{datetime.now()} - Downloaded frames matching pattern: {frames_pattern}")

    def upload_image_stream(self, image_stream: io.BytesIO, object_name: str, bucket_name: str = None, content_type: str = "image/jpeg"):
        image_bytes = image_stream.getbuffer()
        if bucket_name is None:
            bucket_name = self.client.gradcam_frames_bucket
        self.client.upload_image_stream_to_minio(
            bucket_name=bucket_name,
            object_name=object_name,
            data_stream=image_stream,
            length=len(image_bytes),
            content_type=content_type
        )
        print(f"{datetime.now()} - Uploaded image stream to {bucket_name}/{object_name}")

    def delete_frames(self):
        for file_name in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"{datetime.now()} - Deleted all local frames from: {self.download_path}")
