import os
import cv2
import dlib
import glob
import numpy as np
from PIL import Image
from datetime import datetime


class VideoProcessor:
    def __init__(self, frames_path):
        print(f'{datetime.now()} - Creating Instance of Video Processing for {os.path.basename(frames_path)}')
        self.frames_path = frames_path
        self.detector = dlib.get_frontal_face_detector()
        self.original_frames = self.load_original_frames()  # Load original frames

    def load_original_frames(self):
        original_frames = []
        frame_paths = sorted(glob.glob(f'{self.frames_path}*.jpg'))  # Ensure sorted order
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            original_frames.append(np.array(frame))  # Store as NumPy array
        return original_frames

    def extract_faces(self):
        face_images = []
        frames = sorted(glob.glob(f'{self.frames_path}*.jpg'))  # Ensure sorted order

        for idx, frame in enumerate(frames):
            image = Image.open(frame)
            rgb_frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

            # Detect faces using dlib
            print(f'{datetime.now()} - Detecting faces from frame {idx}/{len(frames)}')
            faces = self.detector(rgb_frame)

            # Check if any faces were detected
            if len(faces) == 0:
                print(f'No faces detected in frame {idx}.')
            else:
                for face in faces:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    # Crop face region from the original RGB image using NumPy
                    print(f'{datetime.now()} - Cropping face regions of {os.path.basename(self.frames_path)} - {idx + 1}/{len(frames)}')
                    face_img = rgb_frame[y:y+h, x:x+w]  # Use NumPy array for face image
                    face_images.append((face_img, rgb_frame))  # Store both face and original frame

        return face_images
