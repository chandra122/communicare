# Models Directory

This directory contains trained machine learning models for the Communicare system.

## Required Model Files

### Sign Language Recognition:
- `sign_detection_best.keras` or `sign_detection_best.h5`
- `sign_actions.pkl` (label mapping)

### Fall Detection:
- `fall_detection_best.keras` or `fall_detection_final.keras`
- `fall_detection_best.h5` or `fall_detection_final.h5`
- `fall_actions.pkl` (label mapping)

## File Sizes
- Model files are large (50-200 MB each)
- These files are excluded from git repository due to size limitations
- Use Git LFS if you need to version control models

## How to Obtain Models

1. **Train your own models** using training scripts (not included in repository)
2. **Download from cloud storage** (if provided by author)
3. **Contact author** for model access

## Model Specifications

### Sign Language Model:
- **Input**: 30 frames × 258 features
- **Architecture**: LSTM neural network
- **Output**: 21 sign language gestures
- **Trained on**: 1,037 sequences

### Fall Detection Model:
- **Input**: 30 frames × 51 features
- **Architecture**: LSTM neural network
- **Output**: 2 classes (normal, fall)
- **Trained on**: 126 scenarios

## Usage

Place all model files in this `models/` directory. The detection scripts will automatically locate them.

