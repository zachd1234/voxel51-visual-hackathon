from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image

import torch

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

class CLIPSeg:

    def predict(user_input: str, image: Image) -> torch.Tensor:

        inputs = processor(text=[user_input], images=[image] * 1, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.unsqueeze(1)

        return torch.sigmoid(preds[0][0])