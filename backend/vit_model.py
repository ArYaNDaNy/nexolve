# vit_model.py
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_PATH = "backend/models/vit_finetuned"  # after training

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_vit(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    fake_prob = float(probs[1])  # fake = class 1
    return fake_prob
