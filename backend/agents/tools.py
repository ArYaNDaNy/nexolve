# agents/tools.py
from models.audio import audio_expert

def check_audio_deepfake(file_bytes):
    """
    Tool function to detect deepfakes in audio files.
    """
    print("🎧 Audio Tool Triggered: Analyzing sound waves...")
    
    result = audio_expert.predict(file_bytes)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    # Format a nice string for the Agent (or API) to return
    verdict = result['verdict'].upper()
    conf = round(result['confidence'] * 100, 2)
    
    return f"🛡️ ANALYSIS COMPLETE: The audio is {verdict} ({conf}% confidence)."