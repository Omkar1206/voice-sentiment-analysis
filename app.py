#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ---------------------------------------------------------
# SPEECH ANALYTICS PROJECT - Streamlit Sentiment Detection App
# ---------------------------------------------------------
import streamlit as st
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import tempfile
import os

# Download VADER lexicon if not already
nltk.download('vader_lexicon', quiet=True)

# Title and description
st.title("Speech Sentiment Analysis App")
st.write("Upload an audio file (WAV format), and the app will transcribe it to text and analyze its sentiment.")

# File uploader
audio_file = st.file_uploader("Upload your WAV audio file", type=["wav"])

# Sentiment function
def map_emotion(score):
    if score >= 0.70:
        return 'Very Happy 😊'
    elif score >= 0.30:
        return 'Happy 🙂'
    elif score >= 0.05:
        return 'Satisfied 😌'
    elif score > -0.05:
        return 'Sad 😔'
    elif score > -0.70:
        return 'Very Sad 😢'
    elif score >= -1:
        return 'Angry 😡'
    else:
        return 'Neutral 😐'

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Step 1: Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Step 2: Convert speech to text
    st.write("**Step 1:** Converting audio to text...")
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        st.success("✅ Transcription Successful!")
        st.subheader("Transcribed Text")
        st.write(text)
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
        os.remove(tmp_path)
        st.stop()

    # Step 3: Perform sentiment detection using VADER
    st.write("**Step 2:** Detecting sentiment using VADER...")
    sia = SentimentIntensityAnalyzer()
    sentiment_result = sia.polarity_scores(text)
    compound_score = sentiment_result['compound']

    # Step 4: Map compound score to emotion
    emotion_label = map_emotion(compound_score)
    st.subheader("Detected Sentiment")
    st.write(f"**Emotion:** {emotion_label}")
    st.write(f"**Compound Score:** {compound_score}")

    # Optional: Show sentiment breakdown
    st.write("Sentiment Breakdown")
    st.json(sentiment_result)

    # Cleanup temporary file
    os.remove(tmp_path)

else:
    st.info("Please upload an audio file to begin analysis.")

