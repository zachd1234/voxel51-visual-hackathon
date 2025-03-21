import os
import torch
import clip
from PIL import Image
import numpy as np

class HazardDetector:
    def __init__(self):
        # Define hazard classes
        self.hazard_classes = [
            "Animal hair", "Dog", "Hazardous electrical cords", "Uneven or broken steps/stairs",
            "Loose or uneven carpets/rugs", "Peeling paint", "Chair", "Cat",
            "Sharp edges or objects", "Mold/dampness", "Fire", "Mouse", "Bed bugs"
        ]
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Tokenize the hazard descriptions
        self.text_tokens = clip.tokenize(self.hazard_classes).to(self.device)
        
        # Encode text features (can be done once since they don't change)
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
    
    def detect_hazards(self, image_path, threshold=0.25, top_k=3):
        """
        Detect hazards in an image
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detection (0-1)
            top_k: Number of top predictions to consider
            
        Returns:
            detected_hazards: List of detected hazard names
            binary_array: Binary array indicating which hazards were detected
            scores: Confidence scores for each hazard
        """
        # Load and preprocess the image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            similarity = similarity[0].cpu().numpy()
        
        # Get indices of top matches
        top_indices = similarity.argsort()[-top_k:][::-1]
        
        # Filter by threshold
        filtered_indices = [i for i in top_indices if similarity[i] >= threshold]
        
        # Get detected hazards
        detected_hazards = [self.hazard_classes[i] for i in filtered_indices]
        
        # Create binary array
        binary_array = [1 if i in filtered_indices else 0 for i in range(len(self.hazard_classes))]
        
        return detected_hazards, binary_array, similarity

# Example usage
if __name__ == "__main__":
    detector = HazardDetector()
    
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    
    if os.path.exists(image_path):
        detected_hazards, binary_array, scores = detector.detect_hazards(image_path)
        
        print(f"Analyzed image: {image_path}")
        print(f"Detected hazards: {detected_hazards}")
        print(f"Binary array: {binary_array}")
        
        # Print confidence scores for all classes
        print("\nConfidence scores:")
        for i, hazard in enumerate(detector.hazard_classes):
            print(f"{hazard}: {scores[i]:.4f}")
    else:
        print(f"Image not found: {image_path}")