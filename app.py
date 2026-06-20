import streamlit as st
import tempfile
from datetime import datetime
from predict import predict_sentiment

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Speech Sentiment Analysis",
    page_icon="",
    layout="centered"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("  Project Information")
    st.write("**Model:** CNN + BiLSTM + Attention")
    st.write("**Feature Extraction:** MFCC (40)")
    st.write("**Classes:** Positive, Negative, Neutral")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Deployment:** Streamlit")

# -----------------------------
# Main Header
# -----------------------------
st.markdown("""
<h1 style='text-align:center;'>
  Speech Sentiment Analysis
</h1>

<p style='text-align:center; font-size:18px;'>
Upload an audio file and predict the sentiment using
CNN + BiLSTM + Attention
</p>

<hr>
""", unsafe_allow_html=True)

# -----------------------------
# File Upload
# -----------------------------
audio_file = st.file_uploader(
    "Upload Audio File",
    type=["wav"]
)

# -----------------------------
# Prediction
# -----------------------------
if audio_file is not None:

    st.audio(audio_file)

    st.write("###   Audio Details")
    st.write(f"**Filename:** {audio_file.name}")
    st.write(f"**Size:** {audio_file.size / 1024:.2f} KB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        temp_audio_path = tmp_file.name

    emotion, confidence = predict_sentiment(temp_audio_path)

    st.write("---")
    st.write("##   Prediction Result")

    if emotion.lower() == "positive":
        st.success(f"😊 Prediction: {emotion}")

    elif emotion.lower() == "negative":
        st.error(f"😔 Prediction: {emotion}")

    else:
        st.info(f"😐 Prediction: {emotion}")

    st.info(f"Confidence: {confidence:.2%}")

    st.progress(float(confidence))

    st.caption(
        f"Prediction generated at: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )