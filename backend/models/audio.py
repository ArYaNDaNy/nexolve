# models/audio.py
import io
import librosa
import numpy as np
import torch
from transformers import pipeline

class AudioDeepfakeDetector:
    def __init__(self):
        self.model_name = "MelodyMachine/Deepfake-audio-detection-V2"
        self._pipe = None # We don't load it yet (Lazy Loading)

    @property
    def pipe(self):
        """
        This property ensures the model is only loaded the FIRST time we need it.
        """
        if self._pipe is None:
            print(f"🔌 Loading {self.model_name} from Hugging Face... (This may take a moment)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 Running on: {device.upper()}")
            
            self._pipe = pipeline(
                "audio-classification", 
                model=self.model_name, 
                device=0 if device == "cuda" else -1
            )
        return self._pipe

    def preprocess_bytes(self, audio_bytes):
        """
        Converts raw file bytes into a Numpy array that the model understands.
        """
        try:
            # 1. Wrap bytes in a 'virtual file' so librosa thinks it's reading from disk
            virtual_file = io.BytesIO(audio_bytes)
            
            # 2. Load audio with librosa
            # target_sr=16000 is standard for many Wav2Vec models, 
            # but pipelines often handle resampling automatically. We keep it raw for now.
            audio_array, sampling_rate = librosa.load(virtual_file, sr=16000)
            
            return audio_array, sampling_rate
        except Exception as e:
            print(f"❌ Error processing audio bytes: {e}")
            return None, None

    def predict(self, audio_bytes):
        """
        The main function your Agent will call.
        """
        # 1. Convert bytes to audio array
        audio_array, rate = self.preprocess_bytes(audio_bytes)
        
        if audio_array is None:
            return {"error": "Could not process audio file"}

        # 2. Run Inference
        # HF Pipelines accept a dictionary for raw audio
        predictions = self.pipe({"array": audio_array, "sampling_rate": rate})
        
        # 3. Format the result nicely
        # The model usually returns a list like [{'label': 'real', 'score': 0.99}, ...]
        best_score = predictions[0]
        
        return {
            "verdict": best_score['label'],      # "real" or "fake"
            "confidence": best_score['score'],   # e.g., 0.998
            "all_scores": predictions            # detailed breakdown
        }

# Create a singleton instance so we don't reload the class every time
audio_expert = AudioDeepfakeDetector()