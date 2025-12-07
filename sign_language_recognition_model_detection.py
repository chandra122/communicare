"""
LSTM Sign Detection - Real-Time Inference

This script loads a trained LSTM sign language recognition model and runs
real-time inference on webcam input using MediaPipe Holistic keypoints.
Optionally, it can send email alerts when specific signs are detected.
"""

import cv2  # OpenCV: Computer vision library (Bradski, 2000)
import mediapipe as mp  # MediaPipe: Holistic pose estimation (Lugaresi et al., 2019)
import numpy as np  # NumPy: Numerical computing
import os
import pickle
import time
import smtplib  # Python SMTP library for email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from tensorflow.keras.models import load_model  # TensorFlow/Keras: Deep learning (Abadi et al., 2016)


# CONFIGURATION
# These settings control how the detection system works
# You can adjust these values to change the system's behavior

# Directory where the trained model(s) and label file are stored
# The script will look for model files in this folder
MODEL_DIR = "models"

# Feature configuration: choose whether to use face features
# - If True: uses pose + face + both hands (1662 features total)
# - If False: uses pose + both hands only (258 features total)
# I set this to False because face features aren't needed for sign language
USE_FACE_FEATURES = False

# Set the number of features based on the configuration above
# This must match what the model was trained with
if USE_FACE_FEATURES:
    NUM_KEYPOINTS = 1662
else:
    NUM_KEYPOINTS = 258

# Inference behavior settings
THRESHOLD = 0.5           # Minimum confidence (0.0 to 1.0) to show a detection
                          # Lower values = more detections but more false positives
                          # Higher values = fewer detections but more accurate
PREDICTION_INTERVAL = 3   # Run the model every 3 frames instead of every frame
                          # This saves computation while still being fast enough
SMOOTHING_WINDOW = 5      # Number of consecutive predictions that must agree
                          # This prevents flickering when model is uncertain

# Email alert configuration
# Set EMAIL_ENABLED to False if you don't want email notifications
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"      # Gmail's SMTP server address
SMTP_PORT = 587                     # Port for TLS encryption
EMAIL_SENDER = "youremail@gmail.com"      # Your email address
EMAIL_PASSWORD = "gmailapppassword"       # Gmail App Password (not your regular password)
                                    # Get this from Google Account settings
EMAIL_RECIPIENT = "emergencyemail@gmail.com"    # Where to send alerts
SIGN_EMAIL_COOLDOWN = 10.0         # Wait 10 seconds between emails for the same sign
                                    # This prevents spam if the same sign is detected repeatedly

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# KEYPOINT EXTRACTION & DRAWING


def extract_keypoints(results):
    """
    Building a flat feature vector from MediaPipe Holistic results.
    
    This function extracts pose, face, and hand landmarks from MediaPipe Holistic
    detection results and concatenates them into a single feature vector suitable
    for LSTM sequence modeling.
    
    Reference: MediaPipe Holistic provides 33 pose landmarks, 468 face landmarks,
    and 21 landmarks per hand (Lugaresi et al., 2019; Bazarevsky et al., 2020).

    Input:
        results: MediaPipe Holistic output for a single frame.
                 It may contain pose, face, left hand, and right hand landmarks.

    Output:
        keypoints: 1D NumPy array of shape:
            - (1662,) if USE_FACE_FEATURES is True
              (Pose 132 + Face 1404 + Left Hand 63 + Right Hand 63)
            - (258,)  if USE_FACE_FEATURES is False
              (Pose 132 + Left Hand 63 + Right Hand 63)
    """
    # Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
    pose = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility]
         for lm in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # Face (optional): 468 landmarks × 3 (x, y, z) = 1404
    if USE_FACE_FEATURES:
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark]
        ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    else:
        face = np.array([])

    # Left hand: 21 landmarks × 3 (x, y, z) = 63
    lh = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Right hand: 21 landmarks × 3 (x, y, z) = 63
    rh = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    if USE_FACE_FEATURES:
        return np.concatenate([pose, face, lh, rh])
    else:
        return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    """
    Runs a single frame through the MediaPipe Holistic pipeline.
    
    MediaPipe Holistic provides unified pose, face, and hand landmark detection
    in real-time, making it suitable for sign language recognition applications.
    
    Reference: MediaPipe Holistic (Lugaresi et al., 2019) combines multiple
    perception models for comprehensive body pose estimation.

    Input:
        image:  BGR image as a NumPy array (e.g., one frame from OpenCV VideoCapture).
        model:  An instance of mp.solutions.holistic.Holistic.

    Output:
        processed_image: BGR image with the same content as input, ready for drawing.
        results:         MediaPipe Holistic results object for this frame.
                         It may contain:
                           - results.pose_landmarks
                           - results.face_landmarks
                           - results.left_hand_landmarks
                           - results.right_hand_landmarks
    """
    # MediaPipe expects RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Runs the Holistic model to get the results
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    processed_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return processed_image, results



def draw_styled_landmarks(image, results):
    """
    Draws detected landmarks on the frame with custom colors.

    Input:
        image:   BGR image (NumPy array) that I want to draw on.
        results: MediaPipe Holistic results for this frame. It may include:
                   - results.face_landmarks
                   - results.pose_landmarks
                   - results.left_hand_landmarks
                   - results.right_hand_landmarks

    Output:
        image:   The same BGR image array, modified in-place with landmarks drawn.
    """
    # Face (optional): 468 landmarks × 3 (x, y, z) = 1404
    if USE_FACE_FEATURES:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )

    # Pose skeleton: 33 landmarks × 4 (x, y, z, visibility) = 132
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )

    # Left hand: 21 landmarks × 3 (x, y, z) = 63
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )

    # Right hand: 21 landmarks × 3 (x, y, z) = 63
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

    # Returns the modified image
    return image

# EMAIL ALERTS FUNCTION

def send_email(subject, body):
    """
    Sends a plain-text email using the configured SMTP server.
    
    Uses Python's built-in smtplib library for SMTP communication.
    Supports Gmail and other SMTP servers with TLS encryption.

    Input:
        subject: String, subject line of the email.
        body:    String, plain-text body of the email.

    Uses:
        EMAIL_ENABLED, SMTP_SERVER, SMTP_PORT,
        EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT.

    Output:
        bool:
            True  - email was sent successfully.
            False - email was not sent (disabled or error occurred).
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

        print(f"[EMAIL] Sent: {subject}")
        return True
    except Exception as e:
        print(f"[EMAIL] Error: {e}")
        return False


# REAL-TIME DETECTION LOOP
def realtime_detection():
    """
    Runs the trained LSTM sign language model in real-time on webcam input.
    
    This function implements a complete real-time sign language recognition pipeline
    using LSTM networks for temporal sequence classification. The approach follows
    the methodology described in Rastgoo et al. (2021) for sign language recognition
    using deep learning and temporal modeling.
    
    The LSTM network (Hochreiter & Schmidhuber, 1997) processes sequences of
    MediaPipe keypoints to recognize sign language gestures. Temporal smoothing
    is applied to reduce prediction noise and improve stability.
    
    References:
    - LSTM Networks: Hochreiter & Schmidhuber (1997)
    - Sign Language Recognition: Rastgoo et al. (2021)
    - MediaPipe Holistic: Lugaresi et al. (2019)

    Input:
        Uses:
          - MODEL_DIR: folder containing the trained model and label file
          - Global config flags: USE_FACE_FEATURES, THRESHOLD, PREDICTION_INTERVAL,
            SMOOTHING_WINDOW, SIGN_EMAIL_COOLDOWN

    Output:
        - Opens an OpenCV window showing live video with MediaPipe landmarks,
          predicted signs, confidence scores, and detection history
        - Optionally sends email alerts when signs are detected
        - Prints summary of detected signs when session ends

    Algorithm:
      1. Load trained LSTM model and gesture labels
      2. Initialize webcam and MediaPipe Holistic detector
      3. For each frame:
         a. Extract keypoints using MediaPipe Holistic
         b. Maintain rolling window of last T frames (sequence buffer)
         c. Periodically run sequence through LSTM model
         d. Apply temporal smoothing to predictions
         e. Display results and send alerts if threshold exceeded
      4. Clean up resources on exit
    """

    print("REAL-TIME SIGN DETECTION")

    # List of possible model file names to check
    # The script will try each one until it finds a file that exists
    # This allows the script to work with different model naming conventions
    MODEL_PATHS = [
        os.path.join(MODEL_DIR, 'sign_lstm_final_single.keras'),
        os.path.join(MODEL_DIR, 'sign_lstm_best_single.keras'),
        os.path.join(MODEL_DIR, 'sign_lstm_final_single.h5'),
        os.path.join(MODEL_DIR, 'sign_lstm_best_single.h5'),
        os.path.join(MODEL_DIR, 'sign_lstm_final.keras'),
        os.path.join(MODEL_DIR, 'sign_lstm_best.keras'),
        os.path.join(MODEL_DIR, 'sign_lstm_final.h5'),
        os.path.join(MODEL_DIR, 'sign_lstm_best.h5'),
    ]

    # List of possible label file names to check
    # The label file maps class numbers to sign names (e.g., 0="help", 1="water")
    ACTIONS_PATHS = [
        os.path.join(MODEL_DIR, 'sign_actions_single.pkl'),
        os.path.join(MODEL_DIR, 'sign_actions.pkl'),
    ]

    model = None
    actions = None
    sequence_length = 30  # Default value, will be updated from the model

    # Load the trained LSTM model
    # Try each possible path until we find a model file that exists
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = load_model(model_path)
            # Get the sequence length from the model's input shape
            # Model input shape is (batch, sequence_length, features)
            # Index 1 tells us how many frames the model expects
            sequence_length = model.input_shape[1]
            print(f"Model loaded. Sequence length: {sequence_length}")
            break

    if model is None:
        print("No model found. Train and save a model under MODEL_DIR before running this script.")
        return

    # Load the gesture labels from the pickle file
    # This file contains the list of sign names the model can recognize
    for actions_path in ACTIONS_PATHS:
        if os.path.exists(actions_path):
            with open(actions_path, 'rb') as f:
                actions_data = pickle.load(f)

            # The label file can be saved in different formats
            # Convert it to a numpy array for easy indexing
            if isinstance(actions_data, dict):
                # If it's a dictionary, convert to array
                actions = np.array([actions_data[k] for k in sorted(actions_data.keys())])
            elif isinstance(actions_data, list):
                # If it's already a list, just convert to array
                actions = np.array(actions_data)
            else:
                # Other formats, try to convert to array
                actions = np.array(actions_data)

            print(f"Loaded actions: {actions}")
            break

    if actions is None:
        print("Actions file not found in MODEL_DIR (e.g., sign_actions.pkl).")
        return

    # Initialize the webcam
    # VideoCapture(0) opens the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    # Set camera resolution to 640x480
    # Lower resolution = faster processing, which is good for real-time use
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe Holistic detector
    # This will detect pose, face, and hand landmarks in each frame
    # min_detection_confidence: how confident MediaPipe needs to be to detect landmarks
    # min_tracking_confidence: how confident it needs to be to keep tracking
    # Lower values = more detections but might be less accurate
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # These variables store the current state of detection
    # I keep a rolling window of the last 30 frames of keypoints for the LSTM model
    sequence = []           # Stores the last 30 frames of keypoint data
    
    # I use these buffers to smooth out predictions and avoid flickering
    # The model might predict different signs in consecutive frames, so I average them
    prediction_buffer = []  # Stores the last 5 predicted sign indices
    confidence_buffer = []  # Stores the last 5 confidence scores
    
    # Current detection result that we show on screen
    current_detection = None # The sign that's currently detected (e.g., "help", "water")
    current_confidence = 0.0 # How confident the model is (0.0 to 1.0)

    # Email tracking to prevent spam
    # I track when I last sent an email for each sign, so I don't send too many
    sign_last_email_time = {}  # Maps sign name to timestamp of last email sent
    sign_text_buffer = []      # History of all detected signs with timestamps

    frame_count = 0

    print("\nStarting real-time detection...")
    print("Press 'q' in the OpenCV window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally so it feels like looking in a mirror
        # This makes it easier for users to see themselves naturally
        frame = cv2.flip(frame, 1)

        # Process the frame with MediaPipe to detect hands, pose, and face
        # This extracts all the keypoints we need for sign language recognition
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw the detected landmarks on the frame so users can see what's being tracked
        draw_styled_landmarks(image, results)

        # Convert MediaPipe results into a flat array of numbers
        # This is the format the LSTM model expects (258 features per frame)
        keypoints = extract_keypoints(results)

        # Check if at least one hand is visible in the frame
        # We need hands to be visible to detect sign language gestures
        hands_detected = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )

        frame_count += 1

        # Update the sequence buffer only when hands are visible
        if hands_detected:
            # Add this frame's keypoints to our sequence buffer
            sequence.append(keypoints)
            # Keep only the last 30 frames (remove older frames)
            # This creates a sliding window of the most recent 1 second of video
            sequence = sequence[-sequence_length:]
        else:
            # If hands disappear, clear everything and start fresh
            # This prevents showing old detections when no one is signing
            sequence = []
            prediction_buffer = []
            confidence_buffer = []
            current_detection = None

        # Run the LSTM model every 3 frames instead of every frame
        # This saves computation while still being fast enough for real-time use
        if frame_count % PREDICTION_INTERVAL == 0:
            # Only make a prediction if we have a full sequence (30 frames) and hands are visible
            # The LSTM needs exactly 30 frames to work properly
            if len(sequence) == sequence_length and hands_detected:
                # Run the sequence through the LSTM model to get predictions
                # The model outputs probabilities for each sign class
                # Shape: (1, 30, 258) -> (1, 30 frames, 258 features per frame)
                probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                # Find which sign has the highest probability
                predicted_idx = int(np.argmax(probs))
                predicted_action = actions[predicted_idx]
                confidence = float(probs[predicted_idx])

                # Add this prediction to our smoothing buffers
                # We keep the last 5 predictions to average them out
                prediction_buffer.append(predicted_idx)
                confidence_buffer.append(confidence)
                
                # Keep only the last 5 predictions (remove older ones)
                # This creates a sliding window for smoothing
                prediction_buffer = prediction_buffer[-SMOOTHING_WINDOW:]
                confidence_buffer = confidence_buffer[-SMOOTHING_WINDOW:]

                # Temporal smoothing: only show a detection if the last 5 predictions all agree
                # This prevents flickering when the model is uncertain
                # For example, if predictions go: "help", "water", "help", "help", "help"
                # We won't show anything until we get 5 consistent "help" predictions
                if len(prediction_buffer) >= SMOOTHING_WINDOW:
                    # Check if all 5 recent predictions are the same
                    unique_preds = np.unique(prediction_buffer)
                    if len(unique_preds) == 1:
                        # All predictions agree, so calculate average confidence
                        avg_conf = float(np.mean(confidence_buffer))
                        
                        # Only show detection if confidence is above threshold (0.5)
                        # This filters out low-confidence predictions
                        if avg_conf > THRESHOLD:
                            current_detection = predicted_action
                            current_confidence = avg_conf

                            # Check if we should send an email for this detection
                            # We use a cooldown period to prevent spam
                            now = time.time()
                            last_time = sign_last_email_time.get(current_detection, 0.0)

                            # Only send email if enough time has passed since last email for this sign
                            if now - last_time > SIGN_EMAIL_COOLDOWN:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                sign_text_buffer.append(f"[{timestamp}] {current_detection}")

                                # Some signs are emergencies and need urgent email alerts
                                # Regular signs get normal email subject, emergencies get "URGENT"
                                emergency_signs = [
                                    'emergency', 'help', 'pain', 'hospital',
                                    'ambulance', 'fire', 'heart_attack'
                                ]
                                is_emergency = current_detection.lower() in emergency_signs

                                if is_emergency:
                                    subject = f"URGENT: {current_detection.upper()} Detected"
                                    body = f"{current_detection.upper()} detected at {timestamp}"
                                else:
                                    subject = f"Sign Detected: {current_detection}"
                                    body = f"{current_detection} detected at {timestamp}"

                                send_email(subject, body)
                                sign_last_email_time[current_detection] = now
                        else:
                            # Confidence too low, don't show detection
                            current_detection = None
                    else:
                        # Predictions don't agree, don't show anything yet
                        current_detection = None
                else:
                    # Not enough predictions yet, keep waiting
                    current_detection = None

        # Display detection results on the video frame
        # Show the detected sign name and confidence score

        if current_detection:
            # Display the detected sign name in large text at the top center
            label_text = current_detection.upper()
            font_scale = 1.5
            thickness = 3

            # Calculate text size to center it properly
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            x = (image.shape[1] - text_w) // 2  # Center horizontally
            y = 50  # Position near top

            # Draw a green background rectangle behind the text for better visibility
            cv2.rectangle(
                image,
                (x - 10, y - text_h - 10),
                (x + text_w + 10, y + baseline + 10),
                (0, 255, 0),  # Green color
                thickness=-1  # Filled rectangle
            )

            # Draw the sign name text in black on the green background
            cv2.putText(
                image,
                label_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                thickness,
                cv2.LINE_AA
            )

            # Show confidence percentage below the sign name
            conf_str = f"{current_confidence * 100:.0f}%"
            cv2.putText(
                image,
                conf_str,
                (x + text_w // 2 - 20, y + text_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )

        # Show a history of the last 3 detected signs at the bottom
        # This helps users see what signs were recognized recently
        if sign_text_buffer:
            recent = [s.split("] ")[1] for s in sign_text_buffer[-3:]]
            history_text = "Detected Signs: " + ", ".join(recent)
            cv2.putText(
                image,
                history_text,
                (10, image.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # Yellow text
                1
            )

        # Show how many frames we have in the buffer
        # This helps users understand when the system is ready to make predictions
        # Buffer needs to be 30/30 before predictions start
        status = f"Buffer: {len(sequence)}/{sequence_length}"
        cv2.putText(
            image,
            status,
            (10, image.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1
        )

        # Display the video frame with all overlays
        cv2.imshow("Real-Time Sign Detection", image)

        # Check if user pressed 'q' to quit
        # waitKey(10) waits 10 milliseconds for a key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Clean up: release camera and close all windows
    # This is important to free up resources properly
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

    print("Detection session ended")
    print(f"\nTotal signs detected: {len(sign_text_buffer)}")
    if sign_text_buffer:
        print("\nDetected Signs History:")
        for entry in sign_text_buffer:
            print(f"  {entry}")



# ENTRY POINT: starting the real-time detection
if __name__ == "__main__":
    realtime_detection()


# REFERENCES:
"""
1. MediaPipe Holistic:
   Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines.
   arXiv preprint arXiv:1906.08172. https://arxiv.org/abs/1906.08172
   
   Bazarevsky, V., et al. (2020). BlazePose: On-device Real-time Body Pose tracking.
   arXiv preprint arXiv:2006.10204. https://arxiv.org/abs/2006.10204

2. LSTM Networks:
   Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
   Neural computation, 9(8), 1735-1780.

3. OpenCV:
   Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

4. TensorFlow/Keras:
   Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning.
   In 12th USENIX symposium on operating systems design and implementation (OSDI 16)
   (pp. 265-283).
   
   Chollet, F. (2015). Keras. GitHub repository. https://github.com/fchollet/keras

5. Sign Language Recognition:
   Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey.
   Expert Systems with Applications, 164, 113794.

6. Temporal Sequence Analysis:
   Graves, A. (2012). Supervised sequence labelling with recurrent neural networks.
   Springer Science & Business Media.

7. SMTP Email Protocol:
   Klensin, J. (2008). Simple Mail Transfer Protocol. RFC 5321.
   https://tools.ietf.org/html/rfc5321
"""
