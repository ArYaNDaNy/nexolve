# main.py
import io
import os
import base64
import tempfile
import numpy as np
from PIL import Image
import imageio.v3 as iio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from vit_model import predict_vit


app = FastAPI(title="Deepfake Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def frame_to_base64(frame_rgb):
    """Convert RGB numpy frame → Base64."""
    pil = Image.fromarray(frame_rgb.astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), sample_fps: float = 1.0):
    filename = file.filename.lower()
    content = await file.read()

    # ---------------- IMAGE ----------------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        prob = predict_vit(pil_img)
        label = "FAKE" if prob >= 0.5 else "REAL"

        return {
            "label": label,
            "confidence": round(prob * 100, 2),
            "suspicious_frames": []
        }

    # ---------------- VIDEO ----------------
    elif filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path.write(content)
        temp_path.close()

        try:
            frames = iio.imiter(temp_path.name)

            frame_probs = []
            suspicious_frames = []

            for idx, frame in enumerate(frames):
                if idx > 30:  # limit for hackathon speed
                    break

                pil_img = Image.fromarray(frame.astype("uint8"))
                prob = predict_vit(pil_img)

                frame_probs.append(prob * 100)

                if prob >= 0.7:
                    suspicious_frames.append(frame_to_base64(frame))

            if not frame_probs:
                raise HTTPException(status_code=400, detail="No valid video frames.")

            avg_conf = float(np.mean(frame_probs))
            label = "FAKE" if avg_conf >= 50 else "REAL"

            return {
                "label": label,
                "confidence": round(avg_conf, 2),
                "suspicious_frames": suspicious_frames
            }

        finally:
            try:
                os.unlink(temp_path.name)
            except:
                pass

    else:
        raise HTTPException(status_code=400, detail="Unsupported file")
