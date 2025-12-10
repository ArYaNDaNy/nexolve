# model_tf.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

# Path to model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xception_deepfake_image_5o.h5")

# Xception requires 299x299 input
IMG_SIZE = (299, 299)

def load_keras_model():
    """Loads the pretrained Keras (.h5) Xception deepfake model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
    
    # compile=False avoids missing custom loss issues
    model = load_model(MODEL_PATH, compile=False)
    return model


def preprocess_pil_image(pil_img):
    """Resize and normalize to [0,1] for this model."""
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(model, pil_img):
    x = preprocess_pil_image(pil_img)
    preds = model.predict(x)
    prob = float(preds.flatten()[0])  # already [0..1]
    return prob


    # Handle model output shapes
    if preds.ndim == 2 and preds.shape[1] == 1:
        # Raw logit → convert via sigmoid
        prob = float(tf.math.sigmoid(preds[0, 0]).numpy())

    elif preds.ndim == 2 and preds.shape[1] == 2:
        # Softmax output → take index 1 as FAKE
        prob = float(preds[0, 1])

    elif preds.ndim == 1:
        prob = float(preds[0])

    else:
        prob = float(tf.math.sigmoid(preds.flatten()[0]).numpy())

    return prob
