# import openai
# import json
# import sounddevice as sd
# import numpy as np
# import wave
# from faster_whisper import WhisperModel

# # OpenAI API key
# OPENAI_API_KEY = "sk-proj-pDTVlj97wJCXQmYjMr98T3BlbkFJCYPeGLdPbvAh69CuZc96"

# # Function to analyze the transcript using GPT
# def analyze_transcript(transcript):
#     openai.api_key = OPENAI_API_KEY
    
#     prompt = (f"Based on the transcript: '{transcript}', "
#               f"provide a concise response to the user's query.")

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=100,
#             temperature=0.5,
#             top_p=1.0,
#             n=1
#         )

#         # Extract the content of the response
#         message_content = response['choices'][0]['message']['content'].strip()

#         return message_content

#     except Exception as e:
#         print(f"Error analyzing transcript: {e}")
#         return {"error": str(e)}

# # Function to record audio
# def record_audio(duration, filename):
#     fs = 16000  # Sample rate
#     print("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
#     sd.wait()  # Wait until recording is finished
#     print("Recording finished.")
    
#     # Save as WAV file
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)  # 16-bit PCM
#         wf.setframerate(fs)
#         wf.writeframes(audio.tobytes())

# # Specify the duration of the recording and the output file
# duration = 10  # seconds
# input_audio_file = "./audio/real_time_input_for_speech_to_text_model.wav"

# # Record audio in real-time
# record_audio(duration, input_audio_file)

# # Choose model size
# model_size = "large-v2"

# # Initialize the WhisperModel
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# # Transcribe the audio
# segments, info = model.transcribe(input_audio_file, beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# # Collect the transcription text
# transcription = " ".join([segment.text for segment in segments])
# print("Transcription:", transcription)

# # Analyze the transcript
# concise_response = analyze_transcript(transcription)
# print("Concise Response:", concise_response)


















import openai
import sounddevice as sd
import wave
from faster_whisper import WhisperModel
import joblib
import numpy as np
import librosa
import soundfile

# OpenAI API key
OPENAI_API_KEY = "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"

# Redefine the extract_feature function
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the emotion recognition model
model = joblib.load('./model/saved_model.pkl')

# Function to predict emotion from audio
def predict_emotion(audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# Function to analyze the transcript using GPT
def analyze_transcript(transcript):
    openai.api_key = OPENAI_API_KEY
    
    prompt = (f"Based on the transcript: '{transcript}', "
              f"provide a concise response to the user's query.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5,
            top_p=1.0,
            n=1
        )

        message_content = response['choices'][0]['message']['content'].strip()
        return message_content

    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return {"error": str(e)}

# Function to record audio
def record_audio(duration, filename):
    fs = 16000  # Sample rate
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording finished.")
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

# Main function to handle the Speech-to-Text process
def run_speech_to_text():
    # Specify the duration of the recording and the output file
    duration = 10  # seconds
    input_audio_file = "./audio/real_time_input_for_speech_to_text_model.wav"

    # Record audio in real-time
    record_audio(duration, input_audio_file)

    # Predict emotion from the recorded audio
    predicted_emotion = predict_emotion(input_audio_file)
    print(f"Predicted Emotion: {predicted_emotion}")

    # Initialize the WhisperModel
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe the audio
    segments, info = model.transcribe(input_audio_file, beam_size=5)
    print(f"Detected language '{info.language}' with probability {info.language_probability}")

    transcription = " ".join([segment.text for segment in segments])
    print("Transcription:", transcription)

    # Analyze the transcript using GPT
    concise_response = analyze_transcript(transcription)
    print("Concise Response:", concise_response)

