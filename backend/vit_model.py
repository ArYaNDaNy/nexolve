import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # backend/
MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_vit(pil_image):
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return float(probs[1])
