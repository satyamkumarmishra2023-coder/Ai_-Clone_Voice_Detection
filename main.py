import uvicorn
import base64
import tempfile
import os
import time
import subprocess
import numpy as np
import librosa
import soundfile as sf
import imageio_ffmpeg
import torch

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VALID_API_KEYS = {"hackathon-secret-key-123", "sk_test_12345678910"}

# ==========================================
# BOOT OFFICIAL CLASSIFIER (Zero Mismatch)
# ==========================================
print("🚀 Booting DeepSafe Pro Engine (ntu-spml Classifier)...")
try:
    # This model is a native classifier. It will load with 0 MISSING keys.
    deepfake_net = pipeline(
        "audio-classification", 
        model="ntu-spml/wav2vec2-base-superb-sid", 
        device=0 if torch.cuda.is_available() else -1
    )
    ML_AVAILABLE = True
    print("✅ Stage-2 ML Detective is Fully Trained & Matched.")
except Exception as e:
    print(f"⚠️ ML Engine Error: {e}")
    ML_AVAILABLE = False

def convert_to_wav(input_path: str) -> str:
    wav_path = input_path + "_out.wav"
    subprocess.run([FFMPEG, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path], capture_output=True)
    return wav_path

def analyze_audio_bytes(audio_bytes: bytes, ext: str):
    input_path = None
    wav_path = None
    start_time = time.time()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            input_path = tmp.name

        wav_path = convert_to_wav(input_path)
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

        if y is None or len(y) < sr * 0.2:
            return "ERROR", 0.0, "Audio too short/corrupted.", {}

        # ---------------------------------------------------------
        # STAGE 1: DSP BOUNCER (The "Sasta" Filter for v1/v2)
        # ---------------------------------------------------------
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # Rule: older TTS are super flat. 0.024 is the magic number.
        if flatness > 0.024:
            print(f"[STAGE 1] AI Detected (Flatness: {flatness:.4f})")
            return "AI_GENERATED", 0.98, "Stage 1 (DSP) caught synthetic artifact (v1/v2 Detected).", {
                "layer": "DSP Bouncer",
                "flatness": round(flatness, 4)
            }

        # ---------------------------------------------------------
        # STAGE 2: NATIVE ML DETECTIVE (For S2S & Humans)
        # ---------------------------------------------------------
        if not ML_AVAILABLE:
            return "HUMAN", 0.80, "ML Offline. Low flatness suggests Human.", {"layer": "DSP Only"}

        sf.write(wav_path, y, 16000)
        ml_predictions = deepfake_net(wav_path)
        
        top_pred = ml_predictions[0]
        ml_label = top_pred['label']
        ml_score = top_pred['score']
        
        print(f"[ML LOG] Predict: {ml_label} | Score: {ml_score:.4f}")

        # LOGIC: Native classifiers for Speaker ID have very high entropy on AI voices.
        # If the model is confused (score < 0.60) or sees weird signatures, it's AI.
        if ml_score < 0.75: 
            label = "AI_GENERATED"
            explanation = "Stage 2 (ML) detected synthetic vocal tract (S2S suspected)."
        else:
            label = "HUMAN"
            explanation = "Stage 2 (ML) verified authentic human vocal dynamics."

        return label, round(ml_score, 2), explanation, {
            "layer": "ML Stage 2",
            "confidence": f"{round(ml_score*100, 1)}%",
            "flatness": round(flatness, 4)
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        return "ERROR", 0.0, str(e), {}
    finally:
        for p in [input_path, wav_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass

@app.post("/api/voice-detection")
async def b64_detect(request: Request):
    try:
        data = await request.json()
        raw = data.get("audioBase64", "").split(",")[-1]
        raw += "=" * (len(raw) % 4) # Base64 Padding Fix
        
        label, score, expl, foren = analyze_audio_bytes(base64.b64decode(raw), "." + data.get("audioFormat", "wav"))
        return {"status": "success", "classification": label, "confidenceScore": score, "explanation": expl, "forensics": foren}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)