from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

class clipvit():

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def get_prob(self, img): #'your_jpg.jpg'
        # Load the JPEG image
        image = Image.open(img)

        # Process the text and image inputs
        inputs = self.processor(
            text=["Animal hair", "Dog", "Hazardous electrical cords or wires", "Uneven or broken steps/staircases", "Loose or uneven carpets/rugs", "Peeling Paint", "Chair", "Cat", "Sharp edges or objects", "Mold/dampness", "Fire", "Mouse", "Bed bugs"],
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Get model outputs
        outputs = self.model(**inputs)

        # Extract the image-text similarity scores
        logits_per_image = outputs.logits_per_image

        # Compute the probabilities using softmax
        probs = logits_per_image.softmax(dim=1)

        # Print the probabilities for each label
        return probs

