# Emotion-Aware Conversational AI

## Overview

Emotion-Aware Conversational AI is a Python-based project that integrates speech emotion recognition, speech-to-text, text-to-speech, and conversational AI (using GPT) to create an interactive, emotion-sensitive voice assistant. The system can:
- Recognize emotions from speech using a trained model
- Transcribe speech to text
- Generate text responses using OpenAI's GPT
- Synthesize speech from text responses
- Operate in three modes: Speech-to-Speech, Speech-to-Text, and Text-to-Speech

## Features
- **Speech-to-Speech**: Record user speech, detect emotion, transcribe, generate a GPT response, and reply with synthesized speech.
- **Speech-to-Text**: Record user speech, detect emotion, transcribe, and generate a concise GPT response as text.
- **Text-to-Speech**: Accept user text input, generate a GPT response, and synthesize it to speech.
- **Emotion Recognition**: Uses a neural network trained on the RAVDESS dataset to classify emotions in speech.

## Project Structure
```
main.py                      # Main menu and entry point
Speech_to_Speech.py          # Speech-to-Speech pipeline
Speech_to_Text.py            # Speech-to-Text pipeline
Text_to_Speech.py            # Text-to-Speech pipeline
speech_emotion_training.py   # Model training and testing
requirements.txt             # Python dependencies
model/                       # Trained model and feature extractor
  saved_model.pkl
  extract_feature.pkl
datasets/ravdess_data/       # RAVDESS dataset (audio files)
audio/                       # Input/output audio files
```

## Installation
1. **Clone the repository**
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Download the RAVDESS dataset** and place it in `datasets/ravdess_data/` as structured above.
4. **Train the emotion recognition model** (optional, pre-trained model provided):
   ```powershell
   python speech_emotion_training.py
   ```

## Usage
Run the main menu:
```powershell
python main.py
```
Follow the prompts to select Speech-to-Speech, Speech-to-Text, or Text-to-Speech modes.

## Requirements
- Python 3.10+
- Microphone and speakers for audio input/output
- OpenAI API key (set in the code)

## Key Dependencies
- openai
- faster-whisper
- TTS (Coqui)
- torchaudio, sounddevice, librosa
- scikit-learn, joblib

## Notes
- Ensure your OpenAI API key is set in the relevant files (`Speech_to_Speech.py`, `Speech_to_Text.py`, `Text_to_Speech.py`).
- The project uses the RAVDESS dataset for emotion recognition.
- Pre-trained models are provided in the `model/` directory.

## License
This project is for educational and research purposes.
