import cv2
import os
# import numpy as np
# from torchvision import v2
from PIL import Image
import os
from datetime import datetime
# import torch
import dlib

# Define the transformations
# transforms = v2.Compose(
#     [
#         v2.Resize((224, 224)),
#         v2.RandomHorizontalFlip(p=0.5),
#         v2.ToTensor(),
#         v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ]
# )

class VideoProcessor:
    def __init__(self, video_path, output_dir='frames'):
    # def __init__(self, video_path):
        print(f'{datetime.now()} - Creating Instance of Video Processing for {os.path.basename(video_path)}')
        self.video_path = video_path
        # self.output_dir = output_dir
        self.detector = dlib.get_frontal_face_detector()

        #Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_frames_and_faces(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        face_images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB format for dlib
            print(f'{datetime.now()} - Converting {os.path.basename(self.video_path)} to frames')
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the image to grayscale for dlib processing
            print(f'{datetime.now()} - Converting {os.path.basename(self.video_path)} frames to Grayscale')
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

            # Detect faces using dlib
            print(f'{datetime.now()} - Detecting faces from {os.path.basename(self.video_path)} frames')
            faces = self.detector(gray_frame)

            # Get the total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Check if any faces were detected
            if len(faces) == 0:
                print(f'No faces detected in frame {frame_count}.')
            else:
                for i, face in enumerate(faces):
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    # Crop face region from the original RGB image using Pillow
                    print(f'{datetime.now()} - Cropping face regions of {os.path.basename(self.video_path)} - {i}/{total_frames}')
                    # face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w])
                    face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w]).convert('RGB')  # Convert to RGB format
                    face_images.append(face_img)

                    # Save the cropped face in RGB format
                    output_path = os.path.join(self.output_dir, f'frame_{frame_count}_face_{i}.jpg')
                    face_img.save(output_path)
                    print(f'Saving {i}/{total_frames}')
                    print(f'Saved face image to: {output_path}')

            frame_count += 1

        cap.release()
        return face_images