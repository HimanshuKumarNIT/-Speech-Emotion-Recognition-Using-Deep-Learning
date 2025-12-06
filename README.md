## -Speech-Emotion-Recognition-Using-Deep-Learning
A deep learning project for analyzing and classifying speech audio. Utilizes advanced audio feature extraction (MFCCs, spectral features) combined with a temporal deep learning model to deliver accurate and insightful audio analysis.

## Overview
- Sentiment classification of audio speech (Positive, Neutral, Negative).
- Uses CNN + BiLSTM + Attention for feature extraction and temporal modeling.
- MFCC features extracted from audio for training.

## Dataset
- TRAIN/ : training audio files
- TEST/  : test audio files
- TRAIN.csv : contains filenames and labels

## Features / EDA
- Waveform and Spectrogram visualization
- MFCC heatmaps per class
- Spectral Centroid & Zero-Crossing Rate analysis
- Class distribution analysis

## Model
- CNN layers for local feature extraction
- BiLSTM for temporal dependencies
- Attention layer to focus on important frames
- Dense layers with softmax output for classification

## Usage
- Run `notebook.ipynb`
- Use `predict_sentiment(audio_path)` for predictions
- Model and label encoder saved as `.h5` and `.pkl`
