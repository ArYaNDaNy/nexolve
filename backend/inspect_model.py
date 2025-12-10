from model_tf import load_keras_model, preprocess_pil_image
from PIL import Image
import numpy as np

model = load_keras_model()

# Load any test image
pil = Image.open("./test.png").convert("RGB")  # <-- put a known REAL face here

x = preprocess_pil_image(pil)

print("Input shape:", x.shape)

pred = model.predict(x)
print("Raw prediction:", pred)
print("Prediction shape:", pred.shape)