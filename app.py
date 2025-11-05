# ---------------------------------------------------------
# SPEECH SENTIMENT ANALYZER – WHISPER + VADER + STREAMLIT
# ---------------------------------------------------------
import streamlit as st
import whisper
import tempfile
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)

# Page setup
st.set_page_config(page_title="Speech Sentiment Analyzer", layout="centered")
st.title("Speech Sentiment Analyzer")
st.write(
    "Upload an audio file (WAV/MP3/M4A) — the app will transcribe it using Whisper and analyze the sentiment using VADER."
)

# Load Whisper model (base model is small and accurate enough)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# File uploader
audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

# Emotion mapping function
def map_emotion(score):
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
    elif score >= -1:
        return "Angry"
    else:
        return "Neutral"

# Process audio
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.write("Step 1: Transcribing audio with Whisper...")

    try:
        # Transcribe using Whisper
        result = model.transcribe(tmp_path)
        text = result["text"]

        st.success("Transcription complete!")
        st.subheader("Transcribed Text")
        st.write(text)

        # Sentiment Analysis
        st.write("Step 2: Analyzing sentiment with VADER...")
        sia = SentimentIntensityAnalyzer()
        sentiment_result = sia.polarity_scores(text)
        compound_score = sentiment_result["compound"]
        emotion_label = map_emotion(compound_score)

        st.subheader("Sentiment Result")
        st.write(f"Emotion: **{emotion_label}**")
        st.write(f"Compound Score: `{compound_score}`")

        st.write("Detailed Sentiment Breakdown:")
        st.json(sentiment_result)

    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        os.remove(tmp_path)
else:
    st.info("Please upload an audio file to begin analysis.")
