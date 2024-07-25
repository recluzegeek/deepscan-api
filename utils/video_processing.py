import cv2
import os
from PIL import Image
import os
from datetime import datetime
import dlib


class VideoProcessor:
    def __init__(self, video_path, output_dir='frames'):
        print(f'{datetime.now()} - Creating Instance of Video Processing for {os.path.basename(video_path)}')
        self.video_path = video_path
        self.detector = dlib.get_frontal_face_detector()


    def extract_frames_and_faces(self, output_dir='frames'):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        face_images = []

        print(f'{datetime.now()} - Converting {os.path.basename(self.video_path)} to frames')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # while True and frame_count < total_frames / 30:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to grayscale for dlib processing
            print(f'{datetime.now()} - Converting Frame to RGB - {frame_count}/{total_frames}')
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Log the shape and data type of the RGB image
            # print(f'RGB Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}')

            # Detect faces using dlib
            print(f'{datetime.now()} - Detecting faces from frame {frame_count}/{total_frames}')
            faces = self.detector(rgb_frame)

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
                    print(f'{datetime.now()} - Cropping face regions of {os.path.basename(self.video_path)} - {frame_count}/{total_frames}')
                    face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w])
                    face_images.append(face_img)

                    # Save the cropped face in RGB format
                    # output_path = os.path.join(output_dir, f'frame_{frame_count}_face_{i}.jpg')
                    # face_img.save(output_path)
                    # print(f'Saving {i}/{total_frames}')
                    # print(f'Saved face image to: {output_path}')

            frame_count += 1

        cap.release()
        return face_images