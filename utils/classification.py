import os
import cv2
import numpy as np
import torch
from datetime import datetime
from .video_processing import VideoProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import time

class Classification:
    def __init__(self, model, frames_path, cam):
        start_time = time.time()
        self.model = model
        self.frames_path = frames_path
        self.cam = cam
        self.video_processor = VideoProcessor(self.frames_path)
        
        # Create visualized directory if it doesn't exist
        self.visualized_dir = os.path.join(os.path.dirname(self.frames_path), 'visualized')
        os.makedirs(self.visualized_dir, exist_ok=True)
        
        print(f'{datetime.now()} - Extracting faces...')
        self.face_images_with_original_frames = self.video_processor.extract_faces()
        print(f'{datetime.now()} - Face extraction completed in {time.time() - start_time:.2f} seconds')

        self.input_size = (224, 224)
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)

        print(f'{datetime.now()} - Creating Classification Instance for {os.path.basename(self.frames_path)}')
        print(f'Using device: {self.device}')
        print(f'Initialization completed in {time.time() - start_time:.2f} seconds')

    def infer(self):
        try:
            total_start_time = time.time()
            self.model.eval()
            results = []
            print(f'{datetime.now()} - Running Inference on {os.path.basename(self.frames_path)}')

            batch_size = 8
            face_batches = [self.face_images_with_original_frames[i:i + batch_size] 
                            for i in range(0, len(self.face_images_with_original_frames), batch_size)]

            print(f'{datetime.now()} - Created {len(face_batches)} batches of size {batch_size}')
            
            for batch_idx, batch in enumerate(face_batches):
                batch_start_time = time.time()
                print(f'{datetime.now()} - Processing batch {batch_idx + 1}/{len(face_batches)}')
                input_tensors = []
                
                # Process each face in the batch
                tensor_prep_start = time.time()
                for face_idx, (face_img, rgb_frame, frame_path) in enumerate(batch):
                    try:
                        rgb_img = cv2.resize(face_img, self.input_size)
                        rgb_img = np.float32(rgb_img) / 255
                        input_tensor = preprocess_image(rgb_img, mean=self.mean, std=self.std)
                        input_tensors.append(input_tensor)

                        # Save the cropped face image
                        cropped_face_path = os.path.join(self.visualized_dir, f"cropped_face_{batch_idx}_{face_idx}.jpg")
                        cv2.imwrite(cropped_face_path, face_img)

                    except Exception as e:
                        print(f'{datetime.now()} - Error processing face {face_idx}: {str(e)}')
                        raise
                print(f'{datetime.now()} - Tensor preparation took {time.time() - tensor_prep_start:.2f} seconds')
                try:
                    # Batch processing
                    gradcam_start = time.time()
                    batch_tensor = np.concatenate(input_tensors, axis=0)
                    batch_tensor = torch.from_numpy(batch_tensor).to(self.device)
                    batch_tensor = batch_tensor.requires_grad_(True)
                    
                    with torch.enable_grad():
                        grayscale_cams = self.cam(input_tensor=batch_tensor,
                                                   targets=None,
                                                   eigen_smooth=True,
                                                   aug_smooth=True)
                    print(f'{datetime.now()} - GradCAM processing took {time.time() - gradcam_start:.2f} seconds')

                    # Visualization processing
                    vis_start = time.time()
                    for idx, ((face_img, rgb_frame, frame_path), grayscale_cam) in enumerate(zip(batch, grayscale_cams)):
                        try:
                            x, y, w, h = self.video_processor.face_coordinates[idx]
                            height, width = face_img.shape[:2]
                            
                            # Get original frame name
                            frame_name = os.path.basename(frame_path)
                            base_name = os.path.splitext(frame_name)[0]
                            output_name = f"{base_name}_visualized.jpg"
                            output_path = os.path.join(self.visualized_dir, output_name)
                            
                            # Process and save the visualization
                            image = cv2.rectangle(rgb_frame.copy(), (x, y), (x + w, y + h), (36,255,12), 1)
                            
                            rgb_img = cv2.resize(face_img, self.input_size)
                            rgb_img = np.float32(rgb_img) / 255
                            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                            
                            # Ensure cam_image_resized matches face dimensions exactly
                            face_region = rgb_frame[y:y+h, x:x+w]
                            cam_image_resized = cv2.resize(cam_image, (face_region.shape[1], face_region.shape[0]))

                            shapes = np.zeros_like(image, np.uint8)
                            shapes[y:y+h, x:x+w] = cam_image_resized

                            alpha = 0.4
                            final_image = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)

                            # Save the visualized image
                            cv2.imwrite(output_path, final_image)

                            # Save the GradCAM visualized face image
                            gradcam_face_path = os.path.join(self.visualized_dir, f"gradcam_face_{batch_idx}_{idx}.jpg")
                            cv2.imwrite(gradcam_face_path, cam_image_resized)

                            results.append(self.cam.outputs)
                            
                            print(f'{datetime.now()} - Processed frame {base_name} ({cam_image_resized.shape} -> {face_region.shape})')
                            
                        except Exception as e:
                            print(f'{datetime.now()} - Error processing visualization for frame {idx}: {str(e)}')
                            print(f'Face shape: {face_img.shape}, CAM resize shape: {cam_image_resized.shape if "cam_image_resized" in locals() else "not created"}')
                            print(f'Face region bounds: y:{y}:{y+h}, x:{x}:{x+w}')
                            raise

                    print(f'{datetime.now()} - Visualization processing took {time.time() - vis_start:.2f} seconds')

                except Exception as e:
                    print(f'{datetime.now()} - Error in batch processing: {str(e)}')
                    raise
                print(f'{datetime.now()} - Batch {batch_idx + 1} completed in {time.time() - batch_start_time:.2f} seconds')

                # Clear GPU cache after processing batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_time = time.time() - total_start_time
            print(f'{datetime.now()} - Total processing completed in {total_time:.2f} seconds')
            print(f'Average time per frame: {total_time/len(self.face_images_with_original_frames):.2f} seconds')
            
            return results
        except Exception as e:
            print(f'{datetime.now()} - Fatal error in infer(): {str(e)}')
            print(f'Error type: {type(e).__name__}')
            import traceback
            print(f'Traceback: {traceback.format_exc()}')
            raise