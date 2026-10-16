# AI Voice Clone Detection API

A **FastAPI** service that checks whether a voice recording is a **real human voice** or **AI-generated**. You send an audio file as Base64 in a JSON request, and the API returns a label, a confidence score, an explanation, and some forensic details.

**Recognition:** Finalist, India AI Impact Summit 2026 (HCL GUVI Buildathon) – top ~2% of 40,000+ teams.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [How It Works](#how-it-works)
4. [API Reference](#api-reference)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Run the Server](#run-the-server)
8. [Notes and Limitations](#notes-and-limitations)
9. [Author](#author)

---

## Features

- Accepts audio as **Base64 in a JSON request** (for example MP3 or WAV)
- Converts any input to **16 kHz mono WAV** using FFmpeg
- **Two-stage detection:**
  - Stage 1: a fast signal-processing check using **spectral flatness**
  - Stage 2: a **wav2vec2 audio classifier** from Hugging Face
- Returns the **label, confidence score, explanation and forensic details**
- Cleans up all temporary files after each request
- Rejects audio that is too short or corrupted

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| API | FastAPI, Uvicorn |
| Audio processing | Librosa, SoundFile, NumPy, FFmpeg (via `imageio-ffmpeg`) |
| Machine learning | PyTorch, Hugging Face Transformers (`ntu-spml/wav2vec2-base-superb-sid`) |

## How It Works

```
Base64 audio  →  Decode  →  FFmpeg (16 kHz mono WAV)  →  Stage 1: Spectral flatness  →  Stage 2: wav2vec2 classifier  →  Result
```

1. **Decode and convert:** The Base64 audio is saved to a temporary file and converted to 16 kHz mono WAV with FFmpeg.
2. **Length check:** Audio shorter than 0.2 seconds is rejected.
3. **Stage 1 – DSP check:** The mean spectral flatness of the audio is calculated. If it is above **0.024**, the voice is labelled `AI_GENERATED`, because older text-to-speech voices tend to be very flat. This threshold was chosen by experiment.
4. **Stage 2 – ML check:** If Stage 1 passes, the audio goes to the wav2vec2 classifier. If its top confidence score is **below 0.75**, the voice is labelled `AI_GENERATED`. Otherwise it is labelled `HUMAN`.
5. **If the ML model cannot load,** the API falls back to Stage 1 only and returns `HUMAN` with a lower confidence of 0.80.

## API Reference

### `POST /api/voice-detection`

**Request body (JSON)**

```json
{
  "language": "Hindi",
  "audioFormat": "mp3",
  "audioBase64": ""
}
```

| Field | Description |
|---|---|
| `language` | Language of the speaker. Accepted in the request but not used by the current code. |
| `audioFormat` | File type of the audio, for example `mp3` or `wav`. |
| `audioBase64` | The audio file encoded as Base64. A `data:...;base64,` prefix is allowed. |

**Example request**

```bash
curl -X POST "http://localhost:8080/api/voice-detection" \
  -H "Content-Type: application/json" \
  -d '{"language": "Hindi", "audioFormat": "mp3", "audioBase64": "<your-base64-audio>"}'
```

**Example response (example values)**

```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.91,
  "explanation": "Stage 2 (ML) verified authentic human vocal dynamics.",
  "forensics": {
    "layer": "ML Stage 2",
    "confidence": "91.0%",
    "flatness": 0.0123
  }
}
```

| Response field | Meaning |
|---|---|
| `classification` | `AI_GENERATED`, `HUMAN`, or `ERROR` |
| `confidenceScore` | Score between 0 and 1 |
| `explanation` | Short reason for the result |
| `forensics` | Which stage decided, plus spectral flatness (and confidence for Stage 2) |

**Error response** (HTTP 400 when the request body cannot be processed)

```json
{
  "status": "error",
  "message": "<error message>"
}
```

## Project Structure

```
<project-folder>/
├── main.py              # API, audio conversion and detection logic
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-link>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
fastapi
uvicorn
numpy
librosa
soundfile
imageio-ffmpeg
torch
transformers
```

The first run downloads the Hugging Face model, so an internet connection is needed once.

## Run the Server

```bash
python main.py
```

The API starts at `http://localhost:8080`. FastAPI's interactive docs are at `http://localhost:8080/docs`.

## Notes and Limitations

- The Stage 1 threshold (0.024) was found by experiment. It is not a general rule and may not work for all audio.
- Stage 1 returns a fixed confidence of 0.98. This is not a measured probability.
- The Stage 2 model is a speaker-identification model. The system treats a low confidence score as a sign of AI speech. This is a heuristic and has not been tested at large scale.
- This is a hackathon project and not a certified forensic tool.
- Possible improvements: add API-key checks to the endpoint, restrict CORS origins, fix the Base64 padding step, train or test a dedicated deepfake-detection model, and add automated tests with a labelled dataset.

## Author

**Satyam Kumar Mishra**
BCA student, Tecnia Institute of Advanced Studies (GGSIPU), Delhi
Email: satyamkumarmishra2023@gmail.com
LinkedIn: linkedin.com/in/satyam-kumar-mishra-a28b183a5
