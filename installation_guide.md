# Communicare Setup and Usage Guide

This guide provides step-by-step instructions for setting up and using the Communicare system, a machine learning-powered solution for sign language recognition and fall detection.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Workflow](#project-workflow)
4. [Step 1: Data Collection](#step-1-data-collection)
5. [Step 2: Model Training](#step-2-model-training)
6. [Step 3: Real-Time Detection](#step-3-real-time-detection)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, please ensure you have the following:

- Python 3.8 or higher installed on your system
- A working webcam connected to your computer
- A Google account for accessing Google Colab (required for model training)
- A Gmail account if you plan to use email notifications (optional)
- A stable internet connection for downloading dependencies and accessing Colab

---

## Installation

### Step 1: Clone or Download the Project

If you're cloning from a Git repository:

```bash
git clone <your-repository-url>
cd Communicare
```

Alternatively, navigate to your project directory if you've downloaded it directly:

```bash
cd "path/to/Communicare"
```

### Step 2: Create a Virtual Environment

Creating a virtual environment is recommended to avoid conflicts with other Python projects. To create and activate one:

```bash
# Create virtual environment
python -m venv communicare_env

# Activate virtual environment
# On Windows:
communicare_env\Scripts\activate
# On Mac/Linux:
source communicare_env/bin/activate
```

### Step 3: Install Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

This will install the following key dependencies:
- OpenCV for computer vision operations
- MediaPipe for pose and hand landmark detection
- TensorFlow and Keras for deep learning model training and inference
- Ultralytics for YOLOv8-pose model
- Additional libraries including NumPy, scikit-learn, and matplotlib

### Step 4: YOLOv8-pose Model

The YOLOv8-pose model file (`yolov8n-pose.pt`) will be automatically downloaded the first time you run the fall detection collection script. If you prefer to download it manually, you can obtain it from the Ultralytics releases page.

---

## Project Workflow

The Communicare system follows a three-step workflow:

```
Data Collection → Model Training → Real-Time Detection
```

1. **Data Collection**: Use the provided scripts to record sign language gestures and fall/normal activity sequences using your webcam.

2. **Model Training**: Upload your collected data to Google Colab and train LSTM models using the provided Jupyter notebooks.

3. **Real-Time Detection**: Run the detection scripts with your trained models to perform real-time sign language recognition and fall detection.

---

## Step 1: Data Collection

### A. Sign Language Data Collection

The script `sign_recognition_data_collection.py` is used to collect sign language gesture data.

#### Usage Instructions

1. Run the script:
   ```bash
   python sign_recognition_data_collection.py
   ```

2. Configure face features:
   When prompted, choose whether to include face features in the data collection. For most use cases, selecting "n" (No) is recommended as it speeds up processing without significantly impacting accuracy.

3. Create a new gesture class:
   Press `n` when prompted to create a new action or gesture class. Enter a descriptive name for your gesture, such as "hello", "help", or "thank_you". Use underscores instead of spaces for multi-word gestures.

4. Record sequences:
   Press the `SPACE` key to start recording a sequence. Perform your gesture clearly for approximately one second (30 frames will be captured automatically). The script will save the sequence and prepare for the next recording. For reliable model training, collect 30-50 sequences per gesture.

5. Navigation controls:
   - `SPACE`: Start recording a new sequence
   - `q`: Quit the script

6. Data storage location:
   All collected sequences are saved in the following directory structure:
   ```
   MP_Data/<gesture_name>/<sequence_number>/
   ```

#### Data Collection Tips

- Perform gestures clearly and maintain consistency across recordings
- Ensure adequate lighting in your environment
- Keep your hands fully visible within the camera frame
- Collect multiple sequences (30-50 recommended) for each gesture
- Slightly vary your position and angle between recordings to improve model robustness

---

### B. Fall Detection Data Collection

The script `fall_detection_collection.py` is used to collect fall and normal activity data.

#### Usage Instructions

1. Run the script:
   ```bash
   python fall_detection_collection.py
   ```

2. Choose activity type:
   Press `n` to create a new activity class. Enter the activity name as "fall", "normal", or "other" depending on what you're recording.

3. Recording process:
   Press `SPACE` to arm the system. Position yourself in front of the camera. The countdown will begin automatically once a person is detected in the frame. Perform your activity (either a fall or normal movement). The script will automatically capture and save 30 frames.

4. Navigation controls:
   - `SPACE`: Arm the system and initiate countdown
   - `q`: Quit the script

5. Data storage location:
   All collected sequences are saved in:
   ```
   Fall_Data/<activity_name>/<sequence_number>/
   ```

#### Data Collection Tips

- Collect data for both "fall" and "normal" activity classes
- Vary fall angles and positions to create a diverse dataset
- Include various normal activities such as walking, sitting, and standing
- Collect 50-100 sequences per class for better model performance
- Ensure your full body remains visible within the camera frame throughout the recording

---

## Step 2: Model Training

### Upload Data to Google Drive

Before training, upload your collected data to Google Drive:

1. Upload your data folders:
   - Upload the `MP_Data/` folder to: `Colab Notebooks/Communicare/MP_Data/`
   - Upload the `Fall_Data/` folder to: `Colab Notebooks/Communicare/Fall_Data/`

2. Verify the folder structure:
   ```
   Colab Notebooks/Communicare/
   ├── MP_Data/
   │   ├── hello/
   │   ├── help/
   │   └── ...
   └── Fall_Data/
       ├── fall/
       ├── normal/
       └── ...
   ```

### A. Sign Language Model Training

Use the notebook `sign_language_recognition_model_training.ipynb` for training the sign language recognition model.

1. Open in Google Colab:
   Upload the notebook to Google Colab, or open it directly from Google Drive if you've stored it there.

2. Run all cells:
   Execute all cells in sequence. The notebook will:
   - Mount your Google Drive
   - Load and preprocess your collected data
   - Build the LSTM model architecture
   - Train the model with your data
   - Save the best model to `models/sign_lstm_best.keras`
   - Generate training visualizations and metrics

3. Training outputs:
   After training completes, you'll find:
   - Trained model: `models/sign_lstm_best.keras`
   - Label mapping: `models/sign_actions.pkl`
   - Training visualizations: `Logs/` directory

4. Expected training time:
   Training duration depends on your dataset size and Colab GPU availability. Typically, training takes between 30 minutes and 2 hours.

---

### B. Fall Detection Model Training

Use the notebook `Fall_Detection_Model_Training.ipynb` for training the fall detection model.

1. Open in Google Colab:
   Upload the notebook to Google Colab, or open it directly from Google Drive.

2. Run all cells:
   Execute all cells sequentially. The notebook will:
   - Mount your Google Drive
   - Load and preprocess your fall detection data
   - Build the LSTM model architecture
   - Train the model with your data
   - Save the best model to `models/fall_detection_best.keras`
   - Generate training visualizations and metrics

3. Training outputs:
   After training completes, you'll find:
   - Trained model: `models/fall_detection_best.keras`
   - Label mapping: `models/fall_actions.pkl`
   - Training visualizations: `Logs/` directory

4. Expected training time:
   Training duration varies based on dataset size and Colab GPU availability. Typically, training takes between 20 minutes and 1.5 hours.

---

## Step 3: Real-Time Detection

After training is complete, download your trained models from Google Drive to your local `models/` folder. The repository already includes pre-trained models, so you can skip training if you prefer to use those.

### A. Sign Language Detection

The script `sign_language_recognition_model_detection.py` performs real-time sign language recognition.

1. Configure email notifications (optional):
   If you want to receive email alerts when signs are detected, open the script and update the email settings around lines 46-48:
   ```python
   EMAIL_SENDER = "your_email@gmail.com"
   EMAIL_PASSWORD = "your_app_password"
   EMAIL_RECIPIENT = "recipient@gmail.com"
   ```
   To disable email notifications, set `EMAIL_ENABLED = False`.

2. Run the detection script:
   ```bash
   python sign_language_recognition_model_detection.py
   ```

3. Using the detection system:
   Position yourself in front of your webcam and perform sign language gestures. The detected gesture name will appear on the screen. If email notifications are enabled, alerts will be sent when signs are detected.

4. Controls:
   - `q`: Quit the application

---

### B. Fall Detection

The script `fall_model_detection.py` performs real-time fall detection.

1. Configure email notifications (optional):
   If you want to receive email alerts when falls are detected, open the script and update the email settings around lines 96-98:
   ```python
   EMAIL_SENDER = "your_email@gmail.com"
   EMAIL_PASSWORD = "your_app_password"
   EMAIL_RECIPIENT = "recipient@gmail.com"
   ```
   To disable email notifications, set `EMAIL_ENABLED = False`.

2. Run the detection script:
   ```bash
   python fall_model_detection.py
   ```

3. Using the detection system:
   Position yourself in front of your webcam. The system will continuously monitor for falls. If a fall is detected, an alert will appear on screen. Email notifications will be sent if enabled.

4. Controls:
   - `q`: Quit the application

---

## Troubleshooting

### Common Issues and Solutions

#### Camera Not Opening

**Problem**: The camera fails to initialize when running detection or collection scripts.

**Solution**: 
- Check if another application is currently using your webcam
- Close other video conferencing or camera applications
- Verify that your webcam is properly connected and recognized by your operating system
- Try running the script with administrator privileges if permission issues occur

#### Model File Not Found

**Problem**: The detection script cannot locate the trained model files.

**Solution**: Ensure the following files are present in your `models/` folder:
- `models/sign_lstm_best.keras` (or `.h5` format)
- `models/fall_detection_best.keras` (or `.h5` format)
- `models/sign_actions.pkl`
- `models/fall_actions.pkl`

If these files are missing, either download them from the repository or train new models using the Colab notebooks.

#### Import Errors

**Problem**: Python cannot import required modules when running scripts.

**Solution**: 
- Verify that your virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check that you're using the correct Python interpreter
- Ensure all package versions match those specified in `requirements.txt`

#### Low Detection Accuracy

**Problem**: The detection system produces incorrect or inconsistent results.

**Solution**:
- Collect more training data (aim for 50+ sequences per class)
- Ensure consistent gesture or activity performance during data collection
- Verify adequate lighting conditions
- Check camera positioning and ensure subjects remain within frame
- Retrain the model with improved data quality

#### Email Not Sending

**Problem**: Email notifications are not being sent despite being enabled.

**Solution**:
- Use a Gmail App Password instead of your regular Gmail password (required for Gmail accounts)
- Generate an App Password from your Google Account settings under Security
- Verify SMTP settings: `smtp.gmail.com` on port `587`
- Check that `EMAIL_ENABLED` is set to `True` in the script
- Ensure your firewall or antivirus isn't blocking the connection

#### Colab Training Errors

**Problem**: Errors occur when running training notebooks in Google Colab.

**Solution**:
- Verify that Google Drive is properly mounted in the notebook
- Check that data folders are located in the correct path: `Colab Notebooks/Communicare/`
- Ensure folder structure matches the expected format (class folders containing sequence folders)
- Verify that you have sufficient Google Drive storage space
- Check that the Colab runtime has GPU enabled if you're using GPU acceleration

---

## Project Structure

The typical project structure is as follows:

```
Communicare/
├── MP_Data/                          # Sign language training data
│   ├── hello/
│   ├── help/
│   └── ...
├── Fall_Data/                        # Fall detection training data
│   ├── fall/
│   ├── normal/
│   └── ...
├── models/                           # Trained models
│   ├── sign_lstm_best.keras
│   ├── fall_detection_best.keras
│   ├── sign_actions.pkl
│   └── fall_actions.pkl
├── Logs/                            # Training visualizations
├── sign_recognition_data_collection.py
├── fall_detection_collection.py
├── sign_language_recognition_model_detection.py
├── fall_model_detection.py
├── sign_language_recognition_model_training.ipynb
├── Fall_Detection_Model_Training.ipynb
├── requirements.txt
└── installation_guide.md
```

---

## Getting Started Summary

Once you've completed the installation and setup, you'll be able to:

- Collect sign language and fall detection data using the provided collection scripts
- Train machine learning models using Google Colab notebooks
- Run real-time detection with email alert capabilities

For additional assistance, refer to the troubleshooting section above or review the error messages carefully to identify specific issues.

---

**Author**: Chandra Bollineni  
**Last Updated**: December 7th, 2025  
**Project**: Communicare - Machine Learning-Powered Sign Language Recognition and Fall Detection System
