import os
import sys
from hazard_detector import HazardDetector
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def test_on_phele_dataset(image_dir, num_images=50, save_results=True):
    """
    Test the hazard detector on the PHELE dataset
    
    Args:
        image_dir: Directory containing the test images
        num_images: Number of test images to process
        save_results: Whether to save the results to a file
    """
    # Initialize the hazard detector
    detector = HazardDetector()
    
    # Create a directory for results if saving
    if save_results:
        results_dir = os.path.join(os.path.dirname(image_dir), "hazard_detection_results")
        os.makedirs(results_dir, exist_ok=True)
    
    # Process each test image
    results = []
    for i in range(1, num_images + 1):
        image_path = os.path.join(image_dir, f"test_{i}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"Processing image {i}/{num_images}: {image_path}")
        
        # Detect hazards
        detected_hazards, binary_array, scores = detector.detect_hazards(image_path)
        
        # Store results
        result = {
            "image_path": image_path,
            "detected_hazards": detected_hazards,
            "binary_array": binary_array,
            "scores": scores
        }
        results.append(result)
        
        # Print results
        print(f"Detected hazards: {detected_hazards}")
        
        # Visualize and save results if requested
        if save_results:
            visualize_result(result, detector.hazard_classes, results_dir)
    
    # Save overall statistics if requested
    if save_results:
        save_statistics(results, detector.hazard_classes, results_dir)
    
    return results

def visualize_result(result, hazard_classes, output_dir):
    """Visualize detection results for a single image"""
    image_path = result["image_path"]
    detected_hazards = result["detected_hazards"]
    scores = result["scores"]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display the image
    img = Image.open(image_path)
    ax1.imshow(np.array(img))
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    # Display the hazard scores as a bar chart
    y_pos = np.arange(len(hazard_classes))
    ax2.barh(y_pos, scores, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(hazard_classes)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('Hazard Detection Scores')
    
    # Highlight detected hazards
    for i, hazard in enumerate(hazard_classes):
        if hazard in detected_hazards:
            ax2.get_yticklabels()[i].set_color('red')
            ax2.get_yticklabels()[i].set_weight('bold')
    
    # Add a title with detected hazards
    plt.suptitle(f"Detected Hazards: {', '.join(detected_hazards) if detected_hazards else 'None'}")
    
    # Save the figure
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_results.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_statistics(results, hazard_classes, output_dir):
    """Save overall statistics for all processed images"""
    # Count occurrences of each hazard
    hazard_counts = {hazard: 0 for hazard in hazard_classes}
    for result in results:
        for hazard in result["detected_hazards"]:
            hazard_counts[hazard] += 1
    
    # Create a bar chart of hazard frequencies
    plt.figure(figsize=(12, 8))
    plt.bar(hazard_counts.keys(), hazard_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Frequency of Detected Hazards')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hazard_frequencies.png"))
    plt.close()
    
    # Save text summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Hazard Detection Summary\n")
        f.write("======================\n\n")
        f.write(f"Total images processed: {len(results)}\n\n")
        f.write("Hazard frequencies:\n")
        for hazard, count in hazard_counts.items():
            f.write(f"  {hazard}: {count} ({count/len(results)*100:.1f}%)\n")

if __name__ == "__main__":
    # Get the image directory from command line or use default
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "/Users/zachderhake/Downloads/PHELE Completely Labelled Image Dataset for Physical Hazards of the Elderly Living Environment/PHELE/PHELE/test/images"
    
    # Run the test
    test_on_phele_dataset(image_dir)
    
    print("Testing complete. Results saved to the 'hazard_detection_results' directory.") 