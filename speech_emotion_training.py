# import glob
# import os
# import numpy as np
# import librosa
# import soundfile
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# import joblib

# # Extract features (mfcc, chroma, mel) from a sound file
# def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
#     with soundfile.SoundFile(file_name) as sound_file:
#         X = sound_file.read(dtype="float32")
#         sample_rate = sound_file.samplerate
#         if chroma:
#             stft = np.abs(librosa.stft(X))
#         result = np.array([])
#         if mfcc:
#             mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#             result = np.hstack((result, mfccs))
#         if chroma:
#             chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, chroma))
#         if mel:
#             mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, mel))
#     return result

# # Emotions in the RAVDESS dataset
# emotions = {
#   '01': 'neutral',
#   '02': 'calm',
#   '03': 'happy',
#   '04': 'sad',
#   '05': 'angry',
#   '06': 'fearful',
#   '07': 'disgust',
#   '08': 'surprised'
# }

# # Emotions to observe
# observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# # Load the dataset
# def load_data(test_size=0.2):
#     x, y = [], []
#     for file in glob.glob("./datasets/ravdess_data/Actor_*/*.wav"):
#         file_name = os.path.basename(file)
#         emotion = emotions[file_name.split("-")[2]]
#         if emotion not in observed_emotions:
#             continue
#         feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#         y.append(emotion)
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# # Splitting the dataset
# x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# # Train the model
# model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# model.fit(x_train, y_train)

# # Save the model to a file
# model_path = './model/saved_model.pkl'
# joblib.dump(model, model_path)

# # Save the extract_feature function to a file
# extract_feature_path = './model/extract_feature.pkl'
# joblib.dump(extract_feature, extract_feature_path)








# Testing
import numpy as np
import librosa
import soundfile
import joblib

# Redefine the extract_feature function before loading it
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
            # Corrected the usage of melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the model
model = joblib.load('./model/saved_model.pkl')

# Function to test the model on a new audio file
def predict_emotion(audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(feature)
    return prediction[0]

# Testing the function to ensure everything works
unseen_audio_file = "./datasets/ravdess_data/Actor_01/03-01-05-01-02-01-01.wav"
predicted_emotion = predict_emotion(unseen_audio_file)
print(f"The predicted emotion is: {predicted_emotion}")
