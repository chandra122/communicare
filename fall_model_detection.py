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


# Model file path detection
# I check multiple possible locations for the model file
# This makes the script work on Windows, local folders, or Google Colab
# without needing to edit the code each time
# The script will use the first path that actually exists
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

# Label file path detection
# Same idea as model paths - check multiple locations
# The label file contains the mapping from class numbers to names (e.g., 0="normal", 1="fall")
POSSIBLE_LABEL_PATHS = [
    r'G:\My Drive\Colab Notebooks\Communicare\models\fall_actions.pkl',
    'fall_actions.pkl',
    'models/fall_actions.pkl',
    '/content/drive/My Drive/Colab Notebooks/Communicare/models/fall_actions.pkl',
    '/content/drive/My Drive/Colab Notebooks/Communicare/fall_actions.pkl',
]

# Find the model file by checking each path until we find one that exists
MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# Find the label file the same way
LABELS_PATH = None
for path in POSSIBLE_LABEL_PATHS:
    if os.path.exists(path):
        LABELS_PATH = path
        break

# Sequence and feature configuration
# These values must match what the model was trained with
SEQUENCE_LENGTH = 30        # Number of frames per sequence (30 frames ≈ 1 second at 30 FPS)
NUM_FEATURES = 51           # Features per frame: 17 keypoints × 3 values (x, y, confidence)

# LSTM prediction behavior settings
THRESHOLD = 0.7             # Minimum confidence (0.0 to 1.0) to accept a prediction
                            # 0.7 means we need 70% confidence before showing a detection
                            # Higher values = fewer false alarms but might miss some falls
PREDICTION_INTERVAL = 3     # Run the LSTM model every 3 frames instead of every frame
                            # This saves computation while still being fast enough
SMOOTHING_WINDOW = 5        # Number of predictions to average together
                            # This smooths out confidence scores and prevents flickering

# Fall alert throttling settings
FALL_ALERT_COOLDOWN = 3.0   # Wait 3 seconds between console alert messages
                            # This prevents flooding the console with messages
ALERT_SOUND = False         # Placeholder for future sound alert feature

# Email notification configuration
# Set EMAIL_ENABLED to False if you don't want email notifications
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"      # Gmail's SMTP server address
SMTP_PORT = 587                     # Port for TLS encryption
EMAIL_SENDER = "your_email@gmail.com"       # Your email address (replace this)
EMAIL_PASSWORD = "your_app_password"        # Gmail App Password (not your regular password)
                                            # Get this from Google Account settings
EMAIL_RECIPIENT = "recipient@gmail.com"     # Where to send fall alerts (replace this)
FALL_EMAIL_COOLDOWN = 30.0                  # Wait 30 seconds between email alerts
                                            # This prevents spam if multiple falls are detected


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

# The label file can be saved in different formats
# Handle both dictionary and list formats for flexibility
if isinstance(label_data, dict):
    # If it's a dict like {"fall": 1, "normal": 0}
    # Convert it to {0: "normal", 1: "fall"} so we can look up by index
    actions = {v: k for k, v in label_data.items()}
    print(f"Labels loaded (dict): {list(label_data.keys())}")
elif isinstance(label_data, list):
    # If it's a list like ["normal", "fall"]
    # Convert it to {0: "normal", 1: "fall"} so we can look up by index
    actions = {i: action for i, action in enumerate(label_data)}
    print(f"Labels loaded (list): {label_data}")
else:
    # Unexpected format, but try to use it anyway
    print("Warning: Unexpected label format. Using raw data as actions map.")
    actions = label_data

# Check what sequence length the model actually expects
# Sometimes models are trained with different sequence lengths
# We adjust our SEQUENCE_LENGTH to match what the model needs
model_input_shape = model.input_shape
if model_input_shape is not None and len(model_input_shape) >= 3:
    # Model input shape is (batch, sequence_length, features)
    # Index 1 is the sequence length
    detected_sequence_length = model_input_shape[1]
    if detected_sequence_length != SEQUENCE_LENGTH:
        # Model expects a different sequence length than our default
        print(f"Warning: Model expects {detected_sequence_length} frames "
              f"but SEQUENCE_LENGTH is {SEQUENCE_LENGTH}.")
        print("Using the model's expected sequence length.")
        SEQUENCE_LENGTH = detected_sequence_length

print("\nRuntime configuration:")
print(f"  Sequence length    : {SEQUENCE_LENGTH}")
print(f"  Features per frame : {NUM_FEATURES}")
print(f"  Confidence threshold: {THRESHOLD}")
print(f"  Prediction interval: {PREDICTION_INTERVAL} frames")

# Initialize YOLOv8-pose model and camera

print("\nLoading YOLOv8-pose model...")
# Load the YOLOv8-pose model for body keypoint detection
# This model detects 17 body keypoints (nose, shoulders, hips, knees, etc.)
# The model file will be downloaded automatically if it doesn't exist
pose_model = YOLO('yolov8n-pose.pt')
print("YOLOv8-pose model loaded.")

print("\nOpening camera...")
# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

# Set camera resolution to 640x480 for faster processing
# Lower resolution = faster processing, which is important for real-time use
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera opened.")

print("\nPress 'q' to quit.")

# Real-Time Detection Loop

# These variables store the current state of detection
# I keep a rolling window of the last 30 frames of keypoints for the LSTM model
sequence = []             # Stores the last 30 frames of body keypoint data

# I use a buffer to smooth out confidence scores and avoid flickering
confidence_buffer = []    # Stores the last 5 confidence scores to average them

frame_count = 0           # Counts total frames processed (used for prediction interval)

# Track when we last sent alerts to prevent spam
last_alert_time = 0.0     # Timestamp of last console alert message
last_email_time = 0.0     # Timestamp of last email alert sent

# Current detection result that we show on screen
current_detection = None  # The activity currently detected ('fall' or 'normal')
current_confidence = 0.0  # How confident the model is (0.0 to 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Flip the frame horizontally so it feels like looking in a mirror
    # This makes it easier for users to position themselves naturally
    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Run YOLOv8-pose to detect people and extract body keypoints
    # This gives us 17 keypoints per person (nose, shoulders, hips, knees, etc.)
    results = pose_model(frame, verbose=False)

    # YOLOv8 automatically draws the pose skeleton on the frame
    # We can display this directly to show what the system sees
    annotated_frame = results[0].plot()

    # Count how many people are detected in this frame
    # We only process the first (most confident) person if multiple are detected
    num_people = len(results[0].keypoints.data)

    # Convert YOLOv8 keypoints into the format our LSTM model expects
    # This extracts 51 features: 17 keypoints × 3 values (x, y, confidence)
    keypoints = extract_keypoints(results)

    # Manage the sequence buffer that stores recent frames
    # The LSTM needs exactly 30 frames to make a prediction
    if num_people > 0:
        # Person is visible, so add this frame's keypoints to the buffer
        sequence.append(keypoints)
        # Keep only the last 30 frames (remove older ones)
        # This creates a sliding window of the most recent 1 second of video
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)
    else:
        # No person detected, so slowly clear the buffer
        # We remove one frame at a time instead of clearing all at once
        # This prevents sudden jumps in detection
        if len(sequence) > 0:
            sequence.pop(0)
        # Clear current detection since no one is in frame
        current_detection = None
        current_confidence = 0.0
        confidence_buffer = []

    # Run the LSTM model every 3 frames instead of every frame
    # This saves computation while still being fast enough for real-time use
    # We only run if we have a full sequence (30 frames) ready
    if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
        # Prepare the sequence for the LSTM model
        # Shape: (1, 30, 51) -> (1 batch, 30 frames, 51 features per frame)
        seq_array = np.array([sequence])

        # Run the sequence through the LSTM model to get predictions
        # The model outputs probabilities for each class (fall, normal, etc.)
        preds = model.predict(seq_array, verbose=0)[0]
        predicted_class = int(np.argmax(preds))
        confidence = float(preds[predicted_class])

        # Smooth the confidence scores by averaging the last 5 predictions
        # This prevents flickering when confidence values jump around
        confidence_buffer.append(confidence)
        if len(confidence_buffer) > SMOOTHING_WINDOW:
            confidence_buffer.pop(0)

        # Calculate average confidence from the buffer
        # If buffer is empty, just use the current confidence
        smoothed_confidence = float(np.mean(confidence_buffer)) if confidence_buffer else confidence

        # Only show detection if confidence is above threshold (0.7)
        # This filters out low-confidence predictions that might be wrong
        if smoothed_confidence >= THRESHOLD:
            current_detection = actions.get(predicted_class, str(predicted_class))
            current_confidence = smoothed_confidence
        else:
            # If confidence drops below half the threshold, clear the detection
            # This prevents showing stale detections when confidence is very low
            if smoothed_confidence < THRESHOLD * 0.5:
                current_detection = None
                current_confidence = 0.0

    # Display information on the video frame
    # Show buffer status and person count at the top

    # Display how many frames we have in the buffer and how many people are detected
    # Buffer needs to be 30/30 before predictions start
    status_text = f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH} | People: {num_people}"
    cv2.putText(
        annotated_frame, status_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Show detection result only when a person is present and we have a detection
    # This prevents showing old detections when no one is in frame
    if num_people > 0 and current_detection:
        if current_detection.lower() == 'fall':
            # Fall detected - use red color to indicate danger
            color = (0, 0, 255)  # Red for fall
            alert_text = f"FALL DETECTED! Confidence: {current_confidence:.2f}"

            # Send alerts with cooldown periods to prevent spam
            # Console alerts can be sent more frequently (every 3 seconds)
            # Email alerts are sent less frequently (every 30 seconds)
            now = time.time()
            if now - last_alert_time > FALL_ALERT_COOLDOWN:
                print(f"\n>>> ALERT: {alert_text} <<<")
                last_alert_time = now

                # Check if enough time has passed since last email
                if now - last_email_time > FALL_EMAIL_COOLDOWN:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    subject = "URGENT: Fall Detected"
                    body = f"Fall detected at {timestamp}. Please check immediately."
                    send_email(subject, body)
                    last_email_time = now
        else:
            # Normal activity detected - use green color
            color = (0, 255, 0)  # Green for normal/other
            alert_text = f"Normal Activity - Confidence: {current_confidence:.2f}"

        # Draw a black background rectangle behind the text for better visibility
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        cv2.rectangle(
            annotated_frame,
            (5, 40),
            (text_size[0] + 15, 75),
            (0, 0, 0),  # Black background
            -1  # Filled rectangle
        )

        # Draw the detection text on the frame
        cv2.putText(
            annotated_frame, alert_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3
        )

    elif num_people == 0:
        # No person detected in the frame
        # Prompt user to step into view
        status_text = "No person detected - step into the frame"
        cv2.putText(
            annotated_frame, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2  # Yellow text
        )
    else:
        # Person is visible but we don't have a full sequence yet
        # Or the model isn't confident enough to make a prediction
        status_text = "Monitoring... building sequence"
        cv2.putText(
            annotated_frame, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2  # Yellow text
        )

    # Show the average confidence score at the bottom
    # This helps users understand how confident the system is
    if confidence_buffer:
        conf_text = f"Avg Confidence: {np.mean(confidence_buffer):.2f}"
        cv2.putText(
            annotated_frame, conf_text, (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1  # White text
        )

    # Display the video frame with all overlays
    cv2.imshow('Fall Detection - Real-Time', annotated_frame)

    # Check if user pressed 'q' to quit
    # waitKey(1) waits 1 millisecond for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release camera and close all windows
# This is important to free up resources properly
cap.release()
cv2.destroyAllWindows()

print("Fall detection stopped")
