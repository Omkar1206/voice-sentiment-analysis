# ---------------------------------------------------------
# SPEECH SENTIMENT ANALYZER – STREAMLIT + PYDUB + WHISPER-TIMESTAMPED + NLTK
# ---------------------------------------------------------

import streamlit as st
import whisper_timestamped as whisper
import tempfile
import os
import nltk
import numpy as np
from pydub import AudioSegment
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (first run only)
nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Speech Sentiment Analyzer", layout="centered")
st.title("Speech Sentiment Analyzer")
st.write(
    "Upload a WAV, MP3, or M4A file. The app will transcribe it and detect the sentiment."
)

# ---------------------------------------------------------
# Load Whisper model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")

model = load_model()

# ---------------------------------------------------------
# Helper: Map compound sentiment score to emotion
# ---------------------------------------------------------
def map_emotion(score: float) -> str:
    if score >= 0.70:
        return "Very Happy"
    elif score >= 0.30:
        return "Happy"
    elif score >= 0.05:
        return "Satisfied"
    elif score > -0.05:
        return "Sad"
    elif score > -0.70:
        return "Very Sad"
    else:
        return "Angry"


# ---------------------------------------------------------
# File upload and processing
# ---------------------------------------------------------
audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        # Convert audio to 16kHz mono PCM using pydub
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        os.remove(tmp_path)

        # Transcription
        st.write("Transcribing audio…")
        result = whisper.transcribe(model, samples)
        text = result["text"]

        st.subheader("Transcribed Text")
        st.write(text)

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        emotion = map_emotion(scores["compound"])

        st.subheader("Detected Emotion")
        st.write(f"**{emotion}** (compound={scores['compound']})")

        st.subheader("Detailed Sentiment Scores")
        st.json(scores)

    except Exception as e:
        st.error(f"Error while processing: {e}")

else:
    st.info("Please upload an audio file to begin analysis.")
