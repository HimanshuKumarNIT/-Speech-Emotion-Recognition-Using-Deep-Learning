import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer


# ---------------------------
# Custom Attention Layer
# ---------------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )

        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(
            tf.keras.backend.dot(x, self.W) + self.b
        )

        a = tf.keras.backend.softmax(e, axis=1)

        output = x * a

        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        return config


# ---------------------------
# Load Model & Label Encoder
# ---------------------------
model = load_model(
    "sentiment_cnn_model.h5",
    custom_objects={"Attention": Attention}
)

le = joblib.load("label_encoder.pkl")


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(file_name):
    audio, sample_rate = librosa.load(
        file_name,
        res_type="kaiser_fast"
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfccs.T, axis=0)


# ---------------------------
# Prediction Function
# ---------------------------
def predict_sentiment(audio_path):

    feature = extract_features(audio_path)

    feature = feature.reshape(1, 40, 1)

    prediction = model.predict(feature, verbose=0)

    predicted_label = le.inverse_transform(
        [np.argmax(prediction)]
    )[0]

    confidence = float(np.max(prediction))

    return predicted_label, confidence


# ---------------------------
# Testing
# ---------------------------
if __name__ == "__main__":

    audio_path = "TEST/112.wav"

    emotion, confidence = predict_sentiment(audio_path)

    print(f"Prediction: {emotion}")
    print(f"Confidence: {confidence:.2%}")