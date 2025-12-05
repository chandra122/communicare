"""
Fall Detection - Real-Time Detection

In this script, I run a trained LSTM fall detection model on top of
YOLOv8‑pose keypoints to detect falls in real time from a webcam feed.
"""

import cv2
import numpy as np
import os
import pickle
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# CONFIGURATION


# I list several possible locations where my trained model might live.
# This lets me reuse the same script on Windows, local folders, or Colab
# without editing the code each time.
POSSIBLE_MODEL_PATHS = [
    # Local Windows paths (.keras)
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_detection_best.keras',
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_detection_final.keras',
    # Local Windows paths (.h5)
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_detection_best.h5',
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_detection_final.h5',
    # Current directory
    'fall_detection_best.keras',
    'fall_detection_final.keras',
    'fall_detection_best.h5',
    'fall_detection_final.h5',
    # Relative models directory
    'models/fall_detection_best.keras',
    'models/fall_detection_final.keras',
    'models/fall_detection_best.h5',
    'models/fall_detection_final.h5',
    # Colab models directory
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_detection_best.keras',
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_detection_final.keras',
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_detection_best.h5',
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_detection_final.h5',
    # Colab root (older location)
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_detection_best.keras',
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_detection_final.keras',
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_detection_best.h5',
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_detection_final.h5',
]

# Same idea for the label file
POSSIBLE_LABEL_PATHS = [
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_actions.pkl',
    'fall_actions.pkl',
    'models/fall_actions.pkl',
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_actions.pkl',
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_actions.pkl',
]

# I pick the first model path that actually exists
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# I pick the first label path that exists
LABELS_PATH = None
for path in POSSIBLE_LABEL_PATHS:
    if os.path.exists(path):
        LABELS_PATH = path
        break

# Sequence and feature config (must match training)
SEQUENCE_LENGTH = 30        # Number of frames per sequence
NUM_FEATURES = 51           # 17 keypoints × 3 (x, y, confidence)

# LSTM prediction behavior
THRESHOLD = 0.7             # Confidence threshold to accept a fall/normal prediction
PREDICTION_INTERVAL = 3     # Run the LSTM every N frames
SMOOTHING_WINDOW = 5        # Number of predictions to average for smoothing

# Fall alert throttling
FALL_ALERT_COOLDOWN = 3.0   # Seconds between console alerts
ALERT_SOUND = False         # Placeholder if I later add sound alerts

# Email notification configuration
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "your_email@gmail.com"       # Replace with your email
EMAIL_PASSWORD = "your_app_password"        # App password (not normal password)
EMAIL_RECIPIENT = "recipient@gmail.com"     # Where to send alerts
FALL_EMAIL_COOLDOWN = 30.0                  # Seconds between email alerts for falls


# UTILITY FUNCTIONS

def extract_keypoints(results):
    """
    Convert YOLOv8‑pose output into a flat feature vector for the LSTM.

    Input:
        results: list of YOLO results from pose_model(frame).
                 I only use results[0] (the first image in the batch).

    Output:
        features: np.ndarray of shape (NUM_FEATURES,)
            17 keypoints × 3 values (x, y, confidence).
            If no person is detected, this returns all zeros.
    """
    # No detections: return a zero vector
    if len(results[0].keypoints.data) == 0:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    # Take the first person's keypoints (YOLOv8-pose sorts by confidence)
    keypoints = results[0].keypoints.data[0].cpu().numpy()

    # Flatten to [x1, y1, c1, x2, y2, c2, ...]
    features = keypoints.flatten().astype(np.float32)
    return features


def send_email(subject, body):
    """
    Send a simple text email using the configured SMTP server.

    Input:
        subject: str
            Subject line of the email.
        body: str
            Plain-text body of the email.

    Output:
        bool:
            True if the email was sent successfully.
            False if email is disabled or an error occurred.
    """
    if not EMAIL_ENABLED:
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


# LOAD MODEL AND LABELS

print("FALL DETECTION - REAL-TIME DETECTION")

# Loading the LSTM model
print("\nLoading model...")
if MODEL_PATH is None:
    print("Error: Model file not found in any of these locations:")
    for path in POSSIBLE_MODEL_PATHS:
        print(f"  - {path}")
    print("\nPlace one of these files where the script can find it:")
    print("  - fall_detection_best.keras (or fall_detection_final.keras / .h5)")
    raise SystemExit(1)

model = load_model(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# Loading the label map for the model
print("Loading labels...")
if LABELS_PATH is None:
    print("Error: Labels file not found in any of these locations:")
    for path in POSSIBLE_LABEL_PATHS:
        print(f"  - {path}")
    print("\nPlace fall_actions.pkl in one of the expected locations.")
    raise SystemExit(1)

with open(LABELS_PATH, 'rb') as f:
    label_data = pickle.load(f)

# Supporting both dict and list formats for the label file
if isinstance(label_data, dict):
    # dict like {"fall": 1, "normal": 0} -> invert to {0: "normal", 1: "fall"}
    actions = {v: k for k, v in label_data.items()}
    print(f"Labels loaded (dict): {list(label_data.keys())}")
elif isinstance(label_data, list):
    # list like ["normal", "fall"] -> {0: "normal", 1: "fall"}
    actions = {i: action for i, action in enumerate(label_data)}
    print(f"Labels loaded (list): {label_data}")
else:
    print("Warning: Unexpected label format. Using raw data as actions map.")
    actions = label_data

# Aligning sequence length with whatever the model actually expects
model_input_shape = model.input_shape
if model_input_shape is not None and len(model_input_shape) >= 3:
    detected_sequence_length = model_input_shape[1]
    if detected_sequence_length != SEQUENCE_LENGTH:
        print(f"Warning: Model expects {detected_sequence_length} frames "
              f"but SEQUENCE_LENGTH is {SEQUENCE_LENGTH}.")
        print("Using the model's expected sequence length.")
        SEQUENCE_LENGTH = detected_sequence_length

print("\nRuntime configuration:")
print(f"  Sequence length    : {SEQUENCE_LENGTH}")
print(f"  Features per frame : {NUM_FEATURES}")
print(f"  Confidence threshold: {THRESHOLD}")
print(f"  Prediction interval: {PREDICTION_INTERVAL} frames")

# Initializing YOLOv8-POSE AND CAMERA

print("\nLoading YOLOv8-pose model...")
pose_model = YOLO('yolov8n-pose.pt')
print("YOLOv8-pose model loaded.")

print("\nOpening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera opened.")

print("\nPress 'q' to quit.")

# Real-Time Detection Loop

# Rolling buffers for temporal modeling and smoothing:      
sequence = []             # Last T keypoint vectors
confidence_buffer = []    # Last N confidence scores
frame_count = 0

last_alert_time = 0.0     # Last console alert time
last_email_time = 0.0     # Last email alert time

current_detection = None  # Current label (e.g., 'fall', 'normal')
current_confidence = 0.0  # Smoothed confidence for current_detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Mirroring the frame to make it feel natural
    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Running YOLOv8‑pose
    results = pose_model(frame, verbose=False)

    # YOLO gives me an annotated image I can show directly
    annotated_frame = results[0].plot()

    # Count how many people are detected
    num_people = len(results[0].keypoints.data)

    # Convert detections to my LSTM feature vector
    keypoints = extract_keypoints(results)

    # Managing the temporal sequence buffer:
    if num_people > 0:
        # When at least one person is present, adding this frame’s features
        sequence.append(keypoints)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)
    else:
        # If the person disappears, slowly clearing the buffer
        if len(sequence) > 0:
            sequence.pop(0)
        current_detection = None
        current_confidence = 0.0
        confidence_buffer = []

    # Running LSTM at intervals:
    if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
        seq_array = np.array([sequence])   # Shape: (1, T, F)

        preds = model.predict(seq_array, verbose=0)[0]
        predicted_class = int(np.argmax(preds))
        confidence = float(preds[predicted_class])

        # Keeping a short history of confidences for smoothing:
        confidence_buffer.append(confidence)
        if len(confidence_buffer) > SMOOTHING_WINDOW:
            confidence_buffer.pop(0)

        smoothed_confidence = float(np.mean(confidence_buffer)) if confidence_buffer else confidence

        if smoothed_confidence >= THRESHOLD:
            current_detection = actions.get(predicted_class, str(predicted_class))
            current_confidence = smoothed_confidence
        else:
            # If confidence drops too low, clearing the current detection:
            if smoothed_confidence < THRESHOLD * 0.5:
                current_detection = None
                current_confidence = 0.0

    # Overlaying information on frame:

    # Showing buffer size and how many people are in the frame:
    status_text = f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH} | People: {num_people}"
    cv2.putText(
        annotated_frame, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Showing detection result only when a person is present:
    if num_people > 0 and current_detection:
        if current_detection.lower() == 'fall':
            color = (0, 0, 255)  # Red for fall
            alert_text = f"FALL DETECTED! Confidence: {current_confidence:.2f}"

            # Console and email alerts with cooldown
            now = time.time()
            if now - last_alert_time > FALL_ALERT_COOLDOWN:
                print(f"\n>>> ALERT: {alert_text} <<<")
                last_alert_time = now

                if now - last_email_time > FALL_EMAIL_COOLDOWN:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    subject = "URGENT: Fall Detected"
                    body = f"Fall detected at {timestamp}. Please check immediately."
                    send_email(subject, body)
                    last_email_time = now
        else:
            color = (0, 255, 0)  # Green for normal/other
            alert_text = f"Normal Activity - Confidence: {current_confidence:.2f}"

        # Drawing a black background rectangle behind the text:
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        cv2.rectangle(
            annotated_frame,
            (5, 40),
            (text_size[0] + 15, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated_frame, alert_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3
        )

    elif num_people == 0:
        # No person detected, prompt the user
        status_text = "No person detected - step into the frame"
        cv2.putText(
            annotated_frame, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
    else:
        # Person visible but model is still building a sequence or not confident
        status_text = "Monitoring... building sequence"
        cv2.putText(
            annotated_frame, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

    # Showing current average confidence:
    if confidence_buffer:
        conf_text = f"Avg Confidence: {np.mean(confidence_buffer):.2f}"
        cv2.putText(
            annotated_frame, conf_text, (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    # Displaying the annotated frame:
    cv2.imshow('Fall Detection - Real-Time', annotated_frame)

    # Exiting on 'q':
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleaning up camera and windows:
cap.release()
cv2.destroyAllWindows()

print("Fall detection stopped")
