# Communicare Project Flow

This document outlines the complete workflow of the Communicare system, from data collection to real-time detection.

---

## Overview

The Communicare system follows a three-stage pipeline:

```
Data Collection → Model Training → Real-Time Detection
```

---

## Stage 1: Data Collection

### Sign Language Data Collection

1. **Run the collection script:**
   ```bash
   python sign_recognition_data_collection.py
   ```

2. **Process:**
   - Choose whether to include face features (recommended: No)
   - Enter the gesture class name (e.g., "help", "water", "emergency")
   - Press `SPACE` to record a sequence
   - Perform the gesture for ~1 second (30 frames captured automatically)
   - Repeat for 30-50 sequences per gesture

3. **Output:**
   - Data saved to: `MP_Data/<gesture_name>/<sequence_number>/`
   - Each sequence contains 30 frames of keypoint data (.npy files)

### Fall Detection Data Collection

1. **Run the collection script:**
   ```bash
   python fall_detection_collection.py
   ```

2. **Process:**
   - Enter activity class ("fall", "normal", or "other")
   - Press `SPACE` to arm the system
   - Step into frame (countdown starts automatically)
   - Perform the activity (fall or normal movement)
   - System captures 30 frames automatically

3. **Output:**
   - Data saved to: `Fall_Data/<activity_name>/<sequence_number>/`
   - Each sequence contains 30 frames of pose keypoint data (.npy files)

---

## Stage 2: Model Training

### Upload Data to Google Drive

1. Upload collected data folders to Google Drive:
   - `MP_Data/` → `Colab Notebooks/Communicare/MP_Data/`
   - `Fall_Data/` → `Colab Notebooks/Communicare/Fall_Data/`

### Sign Language Model Training

1. **Open notebook in Google Colab:**
   - `sign_language_recognition_model_training.ipynb`

2. **Process:**
   - Mount Google Drive
   - Load and preprocess collected sequences
   - Build LSTM model architecture
   - Train model (30 minutes - 2 hours)
   - Save best model and label mappings

3. **Output:**
   - Trained model: `models/sign_lstm_best.keras`
   - Label mapping: `models/sign_actions.pkl`
   - Training visualizations: `Logs/` directory

### Fall Detection Model Training

1. **Open notebook in Google Colab:**
   - `Fall_Detection_Model_Training.ipynb`

2. **Process:**
   - Mount Google Drive
   - Load and preprocess collected sequences
   - Build LSTM model architecture
   - Train model (20 minutes - 1.5 hours)
   - Save best model and label mappings

3. **Output:**
   - Trained model: `models/fall_detection_best.keras`
   - Label mapping: `models/fall_actions.pkl`
   - Training visualizations: `Logs/` directory

---

## Stage 3: Real-Time Detection

### Download Models

After training, download the model files from Google Drive to your local `models/` folder.

### Sign Language Detection

1. **Run the detection script:**
   ```bash
   python sign_language_recognition_model_detection.py
   ```

2. **Process:**
   - Script loads trained LSTM model
   - Initializes MediaPipe Holistic for keypoint detection
   - Captures webcam feed
   - Extracts keypoints from each frame
   - Maintains rolling window of 30 frames
   - Runs LSTM model every 3 frames
   - Applies temporal smoothing to predictions
   - Displays detected gesture on screen
   - Sends email alerts (if enabled)

3. **Output:**
   - Real-time video feed with detected gestures
   - Confidence scores
   - Email notifications for detected signs

### Fall Detection

1. **Run the detection script:**
   ```bash
   python fall_model_detection.py
   ```

2. **Process:**
   - Script loads trained LSTM model
   - Initializes YOLOv8-pose for body keypoint detection
   - Captures webcam feed
   - Extracts 17 body keypoints per frame
   - Maintains rolling window of 30 frames
   - Runs LSTM model every 3 frames
   - Applies temporal smoothing to predictions
   - Displays detection results on screen
   - Sends email alerts for detected falls

3. **Output:**
   - Real-time video feed with pose skeleton
   - Fall/normal activity detection
   - Confidence scores
   - Email notifications for detected falls

---

## Technical Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sign Language:                    Fall Detection:          │
│  ┌──────────────────┐            ┌──────────────────┐     │
│  │ Webcam Input      │            │ Webcam Input      │     │
│  │ MediaPipe Holistic│            │ YOLOv8-pose      │     │
│  │ Extract Keypoints │            │ Extract Keypoints │     │
│  │ Save Sequences    │            │ Save Sequences   │     │
│  └──────────────────┘            └──────────────────┘     │
│           │                                │                │
│           └────────────┬───────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│              MP_Data/  &  Fall_Data/                         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Upload data to Google Drive                             │
│  2. Open training notebook in Google Colab                  │
│  3. Load sequences from data folders                        │
│  4. Preprocess data (normalize, split train/val/test)     │
│  5. Build LSTM model architecture                          │
│  6. Train model with callbacks (early stopping, etc.)      │
│  7. Save best model and label mappings                      │
│  8. Generate training visualizations                        │
│                                                              │
│  Output: models/*.keras, models/*.pkl                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 REAL-TIME DETECTION                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sign Language:                    Fall Detection:          │
│  ┌──────────────────┐            ┌──────────────────┐     │
│  │ Load Model       │            │ Load Model        │     │
│  │ MediaPipe        │            │ YOLOv8-pose      │     │
│  │ Webcam Feed      │            │ Webcam Feed      │     │
│  │ Extract Keypoints│            │ Extract Keypoints│     │
│  │ Build Sequence   │            │ Build Sequence   │     │
│  │ LSTM Prediction  │            │ LSTM Prediction  │     │
│  │ Smooth Results   │            │ Smooth Results   │     │
│  │ Display + Email  │            │ Display + Email  │     │
│  └──────────────────┘            └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Data Collection Scripts
- `sign_recognition_data_collection.py` - Collects sign language gesture sequences
- `fall_detection_collection.py` - Collects fall/normal activity sequences

### Training Notebooks
- `sign_language_recognition_model_training.ipynb` - Trains sign language LSTM model
- `Fall_Detection_Model_Training.ipynb` - Trains fall detection LSTM model

### Detection Scripts
- `sign_language_recognition_model_detection.py` - Real-time sign language recognition
- `fall_model_detection.py` - Real-time fall detection

### Supporting Files
- `requirements.txt` - Python dependencies
- `installation_guide.md` - Detailed setup and usage instructions

---

## Data Flow Summary

1. **Raw Video** → MediaPipe/YOLOv8 → **Keypoints** (per frame)
2. **Keypoints** → Sequence Buffer (30 frames) → **Sequence Array**
3. **Sequence Array** → LSTM Model → **Prediction Probabilities**
4. **Probabilities** → Temporal Smoothing → **Final Detection**
5. **Final Detection** → Display + Email Alert → **User Output**

---

## Model Architecture

### Sign Language Model
- **Input:** 30 frames × 258 features (Pose 132 + Left Hand 63 + Right Hand 63)
- **Architecture:** LSTM layers with Dense output
- **Output:** Probability distribution over gesture classes

### Fall Detection Model
- **Input:** 30 frames × 51 features (17 keypoints × 3: x, y, confidence)
- **Architecture:** LSTM layers with Dense output
- **Output:** Probability distribution over activity classes (fall/normal)

---

**Last Updated:** December 6th, 2025  
**Project:** Communicare - Machine Learning-Powered Sign Language Recognition and Fall Detection System

