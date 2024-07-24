# import cv2
# import os
# import numpy as np
# from torchvision import v2
# from PIL import Image
# import torch
# import dlib

# # Define the transformations
# transforms = v2.Compose(
#     [
#         v2.Resize((224, 224)),
#         v2.RandomHorizontalFlip(p=0.5),
#         v2.ToTensor(),
#         v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ]
# )

# class VideoProcessor:
#     def __init__(self, video_path, output_dir='frames'):
#         self.video_path = video_path
#         self.output_dir = output_dir
#         self.detector = dlib.get_frontal_face_detector()  # Initialize dlib's face detector

#         # Create output directory if it doesn't exist
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#     def extract_frames_and_faces(self):
#         cap = cv2.VideoCapture(self.video_path)
#         frame_count = 0
#         face_images = []

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the frame to RGB format for dlib
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Convert the image to grayscale for dlib processing
#             gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

#             # Detect faces using dlib
#             faces = self.detector(gray_frame)

#             # Check if any faces were detected
#             if len(faces) == 0:
#                 print(f'No faces detected in frame {frame_count}.')
#             else:
#                 for i, face in enumerate(faces):
#                     x = face.left()
#                     y = face.top()
#                     w = face.right() - x
#                     h = face.bottom() - y

#                     # Crop face region from the original RGB image using Pillow
#                     face_img = Image.fromarray(rgb_frame[y:y+h, x:x+w])
#                     face_images.append(face_img)

#                     # Save the cropped face in RGB format
#                     output_path = os.path.join(self.output_dir, f'frame_{frame_count}_face_{i}.jpg')
#                     face_img.save(output_path)
#                     print(f'Saved face image to: {output_path}')

#             frame_count += 1

#         cap.release()
#         return face_images

# # # Example usage
# # if __name__ == '__main__':
# #     video_processor = VideoProcessor(video_path='path_to_your_video.mp4', output_dir='output_faces')
# #     face_images = video_processor.extract_frames_and_faces()

# class Classification:
#     def __init__(self, model):
#         self.model = model

#     def preprocess(self, image):
#         return transforms(image)

#     def infer(self, face_images):
#         self.model.eval()  # Set the model to evaluation mode
#         results = []

#         with torch.no_grad():
#             for face in face_images:
#                 # Preprocess the image
#                 input_tensor = self.preprocess(face)
#                 input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

#                 # Perform inference
#                 output = self.model(input_tensor)
#                 results.append(output)

#         return results

# if __name__ == '__main__':
#     # Load your pre-trained Swin Transformer model
#     model = SwinTransformer()  # Replace with your model initialization
#     model.load_state_dict(torch.load('path_to_your_swin_transformer_model.pth'))
#     model.eval()  # Set the model to evaluation mode

#     # Move model to the appropriate device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Initialize video processor
#     video_processor = VideoProcessor(video_path='path_to_your_video.mp4')

#     # Extract faces from the video
#     face_images = video_processor.extract_frames_and_faces()

#     # Initialize classification
#     classifier = Classification(model)

#     # Run inference on the extracted faces
#     inference_results = classifier.infer(face_images)

#     # Process the results as needed
#     for result in inference_results:
#         # Apply softmax or thresholding if necessary
#         probabilities = torch.softmax(result, dim=1)  # Example for multi-class
#         print(probabilities)
