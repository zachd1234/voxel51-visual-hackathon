# YOLO-World Video Object Detection
# Install required packages


import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import time

def process_video(video_path, classes, output_path=None, confidence=0.003):
    # Initialize YOLO-World model (using large model for better accuracy)
    model = YOLOWorld(model_id="yolo_world/l")
    model.set_classes(classes)
    
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
    
    # Initialize annotators
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model.infer(frame, confidence=confidence)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        
        # Create labels with confidence scores
        labels = [
            f"{classes[class_id]} {confidence:0.3f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        
        # Draw annotations
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('YOLO-World Detection', annotated_frame)
        
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
    video_path = "dataset/keeptrack-house-video-with-audio-horizontal-720p test.mov"
    
    # Define the classes you want to detect
    classes = ["green chair"]
    
    output_path = "output.mp4"  # Optional: path to save the processed video
    
    # Process video with low confidence threshold and NMS
    process_video(video_path, classes, output_path, confidence=0.25)