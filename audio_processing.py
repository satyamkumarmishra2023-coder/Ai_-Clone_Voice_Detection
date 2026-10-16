import numpy as np
import librosa

def preprocess(y, sr):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = librosa.util.normalize(y)
    y = librosa.effects.preemphasis(y)
    return y

def extract_features(y, sr):
    y = y[:16000 * 2]  # 2 sec chunk

    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        f0 = f0[~np.isnan(f0)]
        pitch_var = float(np.var(f0)) if len(f0) else 0.0
    except:
        pitch_var = 0.0

    return rms, zcr, flat, centroid, pitch_var

def classify_audio(rms, zcr, flat, centroid, pitch_var):
    score = 0

    if pitch_var < 15:
        score += 2
    if flat > 0.02:
        score += 1
    if zcr > 0.09:
        score += 1
    if rms < 0.01:
        score += 1
    if centroid < 1200:
        score += 1

    if score >= 3:
        return "AI_GENERATED", 0.8
    return "HUMAN", 0.75