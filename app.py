import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os, nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

st.title("Speech Sentiment Analyzer")
st.write("Upload an audio file; the app will transcribe it and analyze the sentiment.")

@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu")

model = load_model()

file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

def map_emotion(score):
    if score >= 0.70: return "Very Happy"
    elif score >= 0.30: return "Happy"
    elif score >= 0.05: return "Satisfied"
    elif score > -0.05: return "Sad"
    elif score > -0.70: return "Very Sad"
    else: return "Angry"

if file:
    st.audio(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name

    st.write("Transcribing...")
    segments, _ = model.transcribe(path)
    text = " ".join([s.text for s in segments])
    os.remove(path)

    st.subheader("Transcribed Text")
    st.write(text)

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    emotion = map_emotion(scores["compound"])

    st.subheader("Detected Emotion")
    st.write(f"{emotion} (compound={scores['compound']})")
    st.json(scores)
else:
    st.info("Please upload an audio file to begin.")
