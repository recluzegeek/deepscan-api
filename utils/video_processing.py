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


    def extract_faces(self, output_dir='frames'):
        
        face_images = []
        # get all video frames
        frames = glob.glob(f'{self.frames_path}*.jpg')

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

                    # Crop face region from the original RGB image using Pillow
                    print(f'{datetime.now()} - Cropping face regions of {os.path.basename(self.frames_path)} - {idx + 1}/{len(frames)}')
                    face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w])
                    face_images.append(face_img)

        return face_images