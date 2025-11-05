import streamlit as st
import whisper_timestamped as whisper
import tempfile, os, nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

st.title("Speech Sentiment Analyzer")
st.write("Upload a WAV or MP3 file to transcribe and analyze sentiment.")

@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")

model = load_model()

def map_emotion(score):
    if score >= 0.70: return "Very Happy"
    elif score >= 0.30: return "Happy"
    elif score >= 0.05: return "Satisfied"
    elif score > -0.05: return "Sad"
    elif score > -0.70: return "Very Sad"
    else: return "Angry"

file = st.file_uploader("Upload audio", type=["wav", "mp3"])
if file:
    st.audio(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name

    st.write("Transcribing…")
    result = whisper.transcribe(model, path)
    text = result["text"]
    os.remove(path)

    st.subheader("Transcribed Text")
    st.write(text)

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    emotion = map_emotion(scores["compound"])
    st.subheader("Detected Emotion")
    st.write(f"{emotion}  (compound={scores['compound']})")
    st.json(scores)
else:
    st.info("Please upload an audio file to begin.")
