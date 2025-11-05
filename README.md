# Speech Sentiment Analyzer – Streamlit App

**Turn voice into insight.**  
This project converts speech into text and performs sentiment analysis using VADER (Valence Aware Dictionary for Sentiment Reasoning).  
It’s built with Streamlit to make speech analytics interactive, accessible, and enjoyable — because data should sound as good as it looks.

---

## Overview

This application takes an audio file, extracts the spoken content using Google Speech Recognition, and analyzes the emotional tone of the text using NLTK’s VADER sentiment analyzer.  
It then classifies the mood behind the voice — from "Very Happy" to "Angry" — and presents it all on a sleek Streamlit dashboard.

---

## Features

- Upload an audio file (supports `.wav` format)  
- Automatic speech-to-text conversion  
- Sentiment analysis using NLTK’s VADER model  
- Emotion mapping into clear categories such as Happy, Sad, Angry, or Neutral  
- Interactive Streamlit interface for instant analysis  
- Lightweight, easy to deploy, and runs locally or online

---

## Project Workflow

1. **Audio Upload** – User uploads a WAV file  
2. **Speech Recognition** – The app transcribes speech using the SpeechRecognition library  
3. **Sentiment Scoring** – VADER generates polarity scores (positive, neutral, negative, compound)  
4. **Emotion Mapping** – Compound score is mapped to an emotion category  
5. **Result Display** – Results are displayed on a Streamlit dashboard with text and score breakdowns

---

## Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **Streamlit** | For interactive web-based UI |
| **SpeechRecognition** | Speech-to-text processing |
| **NLTK (VADER)** | Sentiment analysis engine |
| **Pydub** | Audio format handling |

---

## Installation

Clone the repository and set up dependencies:

```bash
git clone https://github.com/yourusername/speech-sentiment-analyzer.git
cd speech-sentiment-analyzer
pip install -r requirements.txt
