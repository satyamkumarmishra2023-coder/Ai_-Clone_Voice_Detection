AI Voice Clone Detection System

A Python pipeline that tells real human voices apart from AI-generated (cloned) voices. It takes a raw audio file, extracts sound features, and passes them to a classifier that predicts whether the voice is real or fake.

Recognition: Finalist, India AI Impact Summit 2026 (HCL GUVI Buildathon) – top ~2% of 40,000+ teams.

Table of Contents
Features
Tech Stack
How It Works
Project Structure
Installation
Usage
Results
Limitations and Future Work
Author
Features
Reads audio files and converts them into numerical features
Extracts MFCCs, chroma features and zero-crossing rate with Librosa
Handles audio clips of different lengths by turning them into fixed-size inputs
Uses vectorised NumPy operations instead of Python loops for faster feature extraction
Classifies each voice as real or AI-generated with a scikit-learn model
Tech Stack
Area	Tools
Language	Python 3
Audio features	Librosa
Numerical work	NumPy
Machine learning	scikit-learn
How It Works
Raw audio  →  Preprocessing  →  Feature extraction  →  Classifier  →  Real / AI-generated
Preprocessing: Load the audio and convert clips of different lengths into a fixed-size input.
Feature extraction: Compute MFCCs, chroma features and zero-crossing rate, and store them as NumPy arrays.
Classification: The scikit-learn model reads the feature vector and predicts the label.
Project Structure

Replace this with your real folder layout.

<project-folder>/
├── <preprocessing file>.py     # loading and fixed-size conversion
├── <features file>.py          # MFCC, chroma, zero-crossing rate
├── <training file>.py          # trains the classifier
├── <predict file>.py           # predicts on a new audio file
├── requirements.txt
└── README.md
Installation
bash
git clone <your-repo-link>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
Usage

Replace the file names with your real ones.

Train the model:

bash
python <training file>.py

Check a new audio file:

bash
python <predict file>.py --file path/to/audio.wav

Example output:

Prediction: AI-generated   (confidence: <value>)
Results

Add only numbers you really measured.

Item	Value
Dataset used	<name and size>
Model	<model type>
Accuracy on test data	<value>
Feature extraction time per sample	<before> → <after> (after replacing loops with NumPy)
Limitations and Future Work
Performance depends on the data the model was trained on, so new kinds of voice-cloning tools may be harder to detect.
Ideas to improve: more training data, more feature types, and testing on noisy real-world audio.
Author

Satyam Kumar Mishra BCA student, Tecnia Institute of Advanced Studies (GGSIPU), Delhi Email: satyamkumarmishra2023@gmail.com LinkedIn: linkedin.com/in/satyam-kumar-mishra-a28b183a5
