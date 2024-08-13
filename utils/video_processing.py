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
        self.face_coordinates = {}
        self.frames_path = frames_path
        self.detector = dlib.get_frontal_face_detector()
        self.original_frames = self.load_original_frames()

    def load_original_frames(self):
        original_frames = []
        frame_paths = sorted(glob.glob(f'{self.frames_path}*.jpg'))
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            original_frames.append(np.array(frame))
        return original_frames

    def extract_faces(self):
        face_images = []

        for idx, rgb_frame in enumerate(self.original_frames):

            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

            # Face Detection
            print(f'{datetime.now()} - Detecting faces from frame {idx}/{len(self.original_frames)}')
            faces = self.detector(rgb_frame)

            if len(faces) == 0:
                print(f'No faces detected in frame {idx}.')
            else:
                for face in faces:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y
                    # store face coordinates for GRADCAM visualization
                    self.face_coordinates[idx] = [x, y, w, h]

                    print(f'{datetime.now()} - Cropping face regions of {os.path.basename(self.frames_path)} - {idx + 1}/{len(self.original_frames)}')

                    face_img = rgb_frame[y:y+h, x:x+w]
                    face_images.append((face_img,rgb_frame))

        return face_images
