# 🚀 Communicare Setup & Usage Guide

Welcome to **Communicare**! This guide will walk you through setting up and using our Machine Learning-powered Sign Language Recognition and Fall Detection system.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Workflow](#project-workflow)
4. [Step 1: Data Collection](#step-1-data-collection)
5. [Step 2: Model Training](#step-2-model-training)
6. [Step 3: Real-Time Detection](#step-3-real-time-detection)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Webcam** connected and working
- **Google Account** (for Colab training)
- **Gmail Account** (for email notifications - optional)
- **Stable Internet Connection** (for downloading models and Colab)

---

## 💻 Installation

### Step 1: Clone or Download the Project

If you have the project in a repository:
```bash
git clone <your-repository-url>
cd Communicare
```

Or simply navigate to your project directory:
```bash
cd "G:\My Drive\Colab Notebooks\Communicare"
```

### Step 2: Create a Virtual Environment (Recommended)

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

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV (computer vision)
- MediaPipe (pose/hand detection)
- TensorFlow & Keras (deep learning)
- Ultralytics (YOLOv8-pose)
- NumPy, scikit-learn, matplotlib, and more

### Step 4: Download YOLOv8-pose Model (Automatic)

The YOLOv8-pose model (`yolov8n-pose.pt`) will be automatically downloaded on first run. If you want to download it manually:

```bash
# The model will auto-download when you run fall_detection_collection.py
# Or download from: https://github.com/ultralytics/assets/releases
```

---

## 🔄 Project Workflow

Our system follows a **3-step pipeline**:

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Data Collection│  →   │  Model Training  │  →   │ Real-Time       │
│  (Local Scripts) │      │  (Google Colab)  │      │ Detection       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

1. **Collect Data** → Record sign language gestures and fall/normal activities
2. **Train Models** → Upload data to Colab and train LSTM models
3. **Detect in Real-Time** → Run detection scripts with trained models

---

## 📸 Step 1: Data Collection

### A. Sign Language Data Collection

**Script:** `sign_recognition_data_collection.py`

#### How to Use:

1. **Run the script:**
   ```bash
   python sign_recognition_data_collection.py
   ```

2. **Configure face features:**
   - When prompted, choose whether to use face features (y/n)
   - **Recommended:** `n` (No) for faster processing

3. **Create a new gesture class:**
   - Press **`n`** to create a new action/gesture
   - Enter the gesture name (e.g., "hello", "help", "thank_you")

4. **Record sequences:**
   - Press **`SPACE`** to start recording
   - Perform the gesture for ~1 second (30 frames)
   - The script will automatically save the sequence
   - Repeat 30-50 times per gesture for good training data

5. **Navigation:**
   - **`SPACE`**: Start recording a new sequence
   - **`q`**: Quit the script

6. **Data Location:**
   - All sequences are saved in: `MP_Data/<gesture_name>/<sequence_number>/`

#### Tips for Better Data:
- ✅ Perform gestures clearly and consistently
- ✅ Ensure good lighting
- ✅ Keep hands visible in frame
- ✅ Collect 30-50 sequences per gesture
- ✅ Vary your position slightly between recordings

---

### B. Fall Detection Data Collection

**Script:** `fall_detection_collection.py`

#### How to Use:

1. **Run the script:**
   ```bash
   python fall_detection_collection.py
   ```

2. **Choose activity type:**
   - Press **`n`** to create a new activity
   - Enter activity name: `fall`, `normal`, or `other`

3. **Recording process:**
   - Press **`SPACE`** to arm the system
   - Get into position in front of the camera
   - The countdown will start automatically when you're detected
   - Perform the activity (fall or normal movement)
   - The script saves 30 frames automatically

4. **Navigation:**
   - **`SPACE`**: Arm the system and start countdown
   - **`q`**: Quit the script

5. **Data Location:**
   - All sequences are saved in: `Fall_Data/<activity_name>/<sequence_number>/`

#### Tips for Better Data:
- ✅ Collect both "fall" and "normal" activities
- ✅ Vary fall angles and positions
- ✅ Include different normal activities (walking, sitting, standing)
- ✅ Collect 50-100 sequences per class
- ✅ Ensure full body is visible in frame

---

## 🎓 Step 2: Model Training

### Upload Data to Google Drive

1. **Upload your data folders:**
   - Upload `MP_Data/` folder to: `Colab Notebooks/Communicare/MP_Data/`
   - Upload `Fall_Data/` folder to: `Colab Notebooks/Communicare/Fall_Data/`

2. **Verify folder structure:**
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

**Notebook:** `sign_language_recognition_model_training.ipynb`

1. **Open in Google Colab:**
   - Upload the notebook to Google Colab
   - Or open directly from Google Drive

2. **Run all cells:**
   - The notebook will:
     - Mount Google Drive
     - Load and preprocess your data
     - Build and train the LSTM model
     - Save the trained model to `models/sign_lstm_best.keras`
     - Generate training visualizations

3. **Training Output:**
   - Model saved to: `models/sign_lstm_best.keras`
   - Actions list saved to: `models/sign_actions.pkl`
   - Visualizations saved to: `Logs/`

4. **Expected Training Time:**
   - Depends on data size and Colab GPU availability
   - Typically 30 minutes to 2 hours

---

### B. Fall Detection Model Training

**Notebook:** `Fall_Detection_Model_Training.ipynb`

1. **Open in Google Colab:**
   - Upload the notebook to Google Colab
   - Or open directly from Google Drive

2. **Run all cells:**
   - The notebook will:
     - Mount Google Drive
     - Load and preprocess fall detection data
     - Build and train the LSTM model
     - Save the trained model to `models/fall_detection_best.keras`
     - Generate training visualizations

3. **Training Output:**
   - Model saved to: `models/fall_detection_best.keras`
   - Actions list saved to: `models/fall_actions.pkl`
   - Visualizations saved to: `Logs/`

4. **Expected Training Time:**
   - Depends on data size and Colab GPU availability
   - Typically 20 minutes to 1.5 hours

---

## 🎯 Step 3: Real-Time Detection

After training, download your trained models from Google Drive to your local `models/` folder.

### A. Sign Language Detection

**Script:** `sign_language_recognition_model_detection.py`

1. **Configure email (optional):**
   - Open the script and update email settings (lines 46-48):
     ```python
     EMAIL_SENDER = "your_email@gmail.com"
     EMAIL_PASSWORD = "your_app_password"
     EMAIL_RECIPIENT = "recipient@gmail.com"
     ```
   - Or disable email by setting `EMAIL_ENABLED = False`

2. **Run the detection script:**
   ```bash
   python sign_language_recognition_model_detection.py
   ```

3. **Using the detection:**
   - Position yourself in front of the camera
   - Perform sign language gestures
   - The detected gesture will appear on screen
   - Email alerts will be sent for detected signs (if enabled)

4. **Controls:**
   - **`q`**: Quit the application

---

### B. Fall Detection

**Script:** `fall_model_detection.py`

1. **Configure email (optional):**
   - Open the script and update email settings (lines 96-98):
     ```python
     EMAIL_SENDER = "your_email@gmail.com"
     EMAIL_PASSWORD = "your_app_password"
     EMAIL_RECIPIENT = "recipient@gmail.com"
     ```
   - Or disable email by setting `EMAIL_ENABLED = False`

2. **Run the detection script:**
   ```bash
   python fall_model_detection.py
   ```

3. **Using the detection:**
   - Position yourself in front of the camera
   - The system will monitor for falls
   - If a fall is detected, an alert will appear
   - Email notifications will be sent (if enabled)

4. **Controls:**
   - **`q`**: Quit the application

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### ❌ **Camera not opening**
- **Solution:** Check if another application is using the camera
- Close other video apps and try again

#### ❌ **Model file not found**
- **Solution:** Ensure trained models are in the `models/` folder:
  - `models/sign_lstm_best.keras` (or `.h5`)
  - `models/fall_detection_best.keras` (or `.h5`)
  - `models/sign_actions.pkl`
  - `models/fall_actions.pkl`

#### ❌ **Import errors**
- **Solution:** Make sure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

#### ❌ **Low detection accuracy**
- **Solution:**
  - Collect more training data (50+ sequences per class)
  - Ensure consistent gesture/activity performance
  - Check lighting and camera positioning

#### ❌ **Email not sending**
- **Solution:**
  - Use Gmail App Password (not regular password)
  - Enable "Less secure app access" or use App Passwords
  - Check SMTP settings (smtp.gmail.com, port 587)

#### ❌ **Colab training errors**
- **Solution:**
  - Ensure Google Drive is properly mounted
  - Check that data folders are in the correct location
  - Verify folder structure matches expected format

---

## 📁 Project Structure

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
├── models/                           # Trained models (after training)
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
└── SETUP_GUIDE.md                   # This file
```

---

## 🎉 You're All Set!

You now have everything you need to:
- ✅ Collect sign language and fall detection data
- ✅ Train ML models in Google Colab
- ✅ Run real-time detection with email alerts

**Happy detecting!** 🚀

---

## 📞 Need Help?

If you encounter any issues not covered in this guide:
1. Check the error messages carefully
2. Verify all file paths and folder structures
3. Ensure all dependencies are installed
4. Review the troubleshooting section above

---

**Last Updated:** 2024
**Project:** Communicare - ML-Powered Sign Language & Fall Detection System

