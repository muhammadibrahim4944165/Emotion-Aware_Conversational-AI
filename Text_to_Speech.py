# import openai
# import torch
# import torchaudio
# from TTS.api import TTS
# import sounddevice as sd

# # OpenAI API key
# OPENAI_API_KEY = "sk-proj-pDTVlj97wJCXQmYjMr98T3BlbkFJCYPeGLdPbvAh69CuZc96"

# # Initialize TTS model
# tts_model = TTS(model_name='tts_models/en/ljspeech/vits--neon')

# # Function to interact with OpenAI's GPT model
# def get_gpt_response(user_input):
#     openai.api_key = OPENAI_API_KEY
    
#     prompt = (f"User said: '{user_input}'. Respond in a short and concise manner.")
    
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=50,  # Limit the response length
#             temperature=0.5,
#             top_p=1.0,
#             n=1
#         )

#         # Extract the content of the response
#         message_content = response['choices'][0]['message']['content'].strip()

#         return message_content

#     except Exception as e:
#         print(f"Error getting GPT response: {e}")
#         return "Sorry, I couldn't process your request."

# # Function to convert text to speech
# def text_to_speech(text, file_path):
#     tts_model.tts_to_file(text=text, file_path=file_path)
#     return file_path

# # Function to play audio file
# def play_audio(file_path):
#     # Load audio file
#     waveform, sample_rate = torchaudio.load(file_path)

#     # Convert waveform to numpy array for sounddevice
#     waveform = waveform.numpy().T

#     # Play audio
#     sd.play(waveform, sample_rate)
#     sd.wait()  # Wait until audio is done playing

# # Main function to run the process
# def main():
#     user_input = input("Please enter your query: ")
#     gpt_response = get_gpt_response(user_input)
#     print(f"GPT Response: {gpt_response}")
    
#     audio_path = "./audio/output_GPT.wav"
#     text_to_speech(gpt_response, audio_path)
#     play_audio(audio_path)

# # Run the main function
# if __name__ == "__main__":
#     main()




















import openai
import torchaudio
from TTS.api import TTS
import sounddevice as sd

# OpenAI API key
OPENAI_API_KEY = "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"

# Initialize TTS model
tts_model = TTS(model_name='tts_models/en/ljspeech/vits--neon')

# Function to interact with OpenAI's GPT model
def get_gpt_response(user_input):
    openai.api_key = OPENAI_API_KEY
    
    prompt = (f"User said: '{user_input}'. Respond in a short and concise manner.")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.5,
            top_p=1.0,
            n=1
        )

        message_content = response['choices'][0]['message']['content'].strip()
        return message_content

    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return "Sorry, I couldn't process your request."

# Function to convert text to speech
def text_to_speech(text, file_path):
    tts_model.tts_to_file(text=text, file_path=file_path)
    return file_path

# Function to play audio file
def play_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.numpy().T
    sd.play(waveform, sample_rate)
    sd.wait()

# Main logic of the script, exposed as a function
def run_text_to_speech():
    user_input = input("Please enter your query: ")
    gpt_response = get_gpt_response(user_input)
    print(f"GPT Response: {gpt_response}")
    
    audio_path = "./audio/output_GPT.wav"
    text_to_speech(gpt_response, audio_path)
    play_audio(audio_path)
