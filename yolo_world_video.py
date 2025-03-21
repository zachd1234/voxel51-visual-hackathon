# YOLO-World Video Object Detection

import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import time
import torch  # Add torch import to check device availability

def process_video(video_path, classes, output_path=None, confidence=0.003):
    # Check if MPS is available (for Apple Silicon)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize YOLO-World model with MPS support
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
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model.infer(frame, confidence=confidence, device=device)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.2)
        
        # Create labels with confidence scores
        labels = [
            f"{classes[class_id]} {conf:0.3f}"  # Changed confidence to conf to avoid name conflict
            for class_id, conf
            in zip(detections.class_id, detections.confidence)
        ]
        
        # Draw annotations
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
        
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
    video_path = "dataset/keeptrack-house-video-with-audio-horizontal-720p test3.mov"
    
    classes = ["blue pillow"]
    
    output_path = "output.mp4"
    
    confidence = 0.2
    
    process_video(video_path, classes, output_path, confidence)