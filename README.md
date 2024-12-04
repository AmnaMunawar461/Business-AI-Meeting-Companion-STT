# Business-AI-Meeting-Companion-STT

In our project, we'll use OpenAI's Whisper to transform speech into text. Next, we'll use IBM Watson's AI to summarize and find key points. We'll make an app with Hugging Face Gradio as the user interface.

Commands to run:
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env


# installing required libraries in my_env
pip install transformers==4.35.2 torch==2.1.1 gradio==4.44.0 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.19.4

sudo apt update


sudo apt install ffmpeg -y

pip install -U huggingface_hub

import torch
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s = 30 
)

sample = 'audio.mp3'

prediction_text = pipe(sample, batch_size = 8)["text"]
print(prediction_text)

import torch
from transformers import pipeline
import gradio as gr

def transcript_audio(audio_file):
    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s = 30 
)

    prediction_text = pipe(sample, batch_size = 8)["text"]
    return prediction_text

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

interf = gr.Interface(fn=transcript_audio,
inputs=audio_input, outputs=output_text,
title="Audio Transcription App",
description="Upload the audio file"
)
interf.launch(server_name = "0.0.0.0", server_port = 8080)

