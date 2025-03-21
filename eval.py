from ViTL_test import clipvit

c = clipvit()


def evaluate_accuracy(image_paths):
    """
    Evaluates the accuracy of predictions over a list of image paths.

    Parameters:
        image_paths (list of str): List of file paths to JPEG images.

    Returns:
        accuracy (float): The proportion of images where the predicted class matches the ground truth.
    """
    correct = 0
    total = 0

    for path in image_paths:
        # Get predicted probabilities and ground truth one-hot vector for the image.
        probs, ground_truth = get_probs_and_ground_truth(path)

        # Determine the predicted and true classes by taking the argmax.
        predicted_class = np.argmax(probs)
        true_class = np.argmax(ground_truth)

        if predicted_class == true_class:
            correct += 1

        total += 1

    # Compute overall accuracy.
    accuracy = correct / total if total > 0 else 0
    return accuracy


# Example usage:
if __name__ == '__main__':
    # List of JPEG paths (replace these with your actual image paths)
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

    accuracy = evaluate_accuracy(image_paths)
    print(f"Accuracy: {accuracy:.2%}")
