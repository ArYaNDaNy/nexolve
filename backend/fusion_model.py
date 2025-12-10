# fusion_model.py
from sklearn.metrics import classification_report
import numpy as np
from vit_model import predict_vit
from heuristics import texture_score, color_anomaly, boundary_artifact, eye_glint_score
from PIL import Image
import cv2

def run_fusion(pil_img):
    img = np.array(pil_img)

    # 1 — ViT prediction
    vit_prob = predict_vit(pil_img)

    # 2 — Heuristics
    tex = texture_score(img)
    col = color_anomaly(img)
    bound = boundary_artifact(img)
    glint = eye_glint_score(img)

    # 3 — Weighted fusion score
    final_score = (
        0.55 * vit_prob +
        0.15 * (1 - tex) +      # low texture → fake
        0.10 * col +            # more color anomaly → fake
        0.10 * bound +          # boundary artifacts → fake
        0.10 * (1 - glint)      # less eye reflection → fake
    )

    final_score = float(np.clip(final_score, 0, 1))

    return final_score, {
        "vit": vit_prob,
        "texture": tex,
        "color": col,
        "boundary": bound,
        "eye_glint": glint
    }


