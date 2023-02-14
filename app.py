import torch
import whisper
import pytube
import librosa
import streamlit as st
import numpy as np
from fpdf import FPDF



def predict(url=None, model = "medium", translation="No",tran_lang="en"):

    model_m = whisper.load_model(model)

    file_path = 'https://cf-courses-data.s3.ujs.cloud-object-storage.appdomain.cloud/IBM-GPXX0EPMEN/20220627_140242.mp4'

    audio_35 = whisper.load_audio(file_path)

    audio = whisper.pad_or_trim(audio_35)

    mel = whisper.log_mel_spectrogram(audio).to(model_m.device)

    _,probs = model_m.detect_language(mel)

    lang_dict = sorted(probs)

    transcription = model_m.transcribe(file_path,fp16=False)["text"]

    if translation == "Yes":
        trans = model_m.transcribe(file_path,language=tran_lang,fp16=False)["text"]
        return lang_dict, transcription, trans
    else:
        return lang_dict, transcription

url = st.text_input(value="Please enter the YouTube url: ", label="YouTube URL")
tran_req = st.selectbox(label="Do you want to translate the transcript?",options=("Yes","No"))

if tran_req=="Yes":
    lang = st.selectbox(label="Please select the required language: ", options=("en","fr","jp","bn","de","ga"))
else:
    lang = "en"

if st.button("Generate"):
    lang_d,transcription,trans = predict(url,translation=tran_req,tran_lang=lang)

    st.write(lang_d)
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size = 15)
    
    for words in transcription:
        pdf.cell(200, 10, txt = x, ln = 1, align = 'L')
    output = pdf.output("transcript.pdf")
    st.download(label="Click here to download the transcript", data=output, mime='pdf',file_name="transcript.pdf")



