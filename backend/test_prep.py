from model_tf import load_keras_model
from PIL import Image
import numpy as np
import tensorflow as tf

model = load_keras_model()

pil = Image.open("test.png").convert("RGB")
pil = pil.resize((299, 299))

arr_raw = np.expand_dims(np.array(pil).astype("float32"), 0)

arr_norm = arr_raw / 255.0

from tensorflow.keras.applications.xception import preprocess_input
arr_xcep = preprocess_input(arr_raw.copy())

print("RAW:", model.predict(arr_raw))
print("NORM:", model.predict(arr_norm))
print("XCEPTION:", model.predict(arr_xcep))