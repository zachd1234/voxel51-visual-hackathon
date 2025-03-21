# YOLO-World Video Object Detection
# This script demonstrates how to use YOLO-World for real-time object detection in videos.

# Install required packages
# !pip install ultralytics opencv-python-headless

import cv2
from ultralytics import YOLO
import numpy as np
import time

# Initialize YOLO-World Model
model = YOLO('yolov8s-world.pt')

def process_video(video_path, output_path=None):
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection using standard YOLO predict
        results = model.predict(source=frame, conf=0.25)
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('YOLO Detection', annotated_frame)
        
        # Write frame if output path is provided
        if writer:
            writer.write(annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_path = "dataset/keeptrack-house-video-with-audio-horizontal-720p.mov"  # Replace with your video path
    output_path = "output.mp4"  # Optional: path to save the processed video
    
    process_video(video_path, output_path) 