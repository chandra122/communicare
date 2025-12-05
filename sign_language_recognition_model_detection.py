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

# Directory where the trained model(s) and label file are stored
MODEL_DIR = "models"

# Feature configuration:
# - If True: using pose + face + both hands (1662 features)
# - If False: using pose + both hands only (258 features)
USE_FACE_FEATURES = False

if USE_FACE_FEATURES:
    NUM_KEYPOINTS = 1662
else:
    NUM_KEYPOINTS = 258

# Inference behavior
THRESHOLD = 0.5           # Minimum confidence to accept a prediction
PREDICTION_INTERVAL = 3   # Running the model every N frames to save compute
SMOOTHING_WINDOW = 5      # Number of consistent predictions required

# Email alert configuration
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "bollinenichandrasekhar11@gmail.com"      # Email Address of the sender
EMAIL_PASSWORD = "Chandra@11"       # App password if using Gmail
EMAIL_RECIPIENT = "bollinenichandrasekhar11@gmail.com"    # Where alerts go Emergency Alerts
SIGN_EMAIL_COOLDOWN = 10.0                 # Seconds between alerts for same sign

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

    # Candidates for model filenames so I can reuse this script: sign_lstm_final_single.keras, sign_lstm_best_single.keras, sign_lstm_final_single.h5, sign_lstm_best_single.h5, sign_lstm_final.keras, sign_lstm_best.keras, sign_lstm_final.h5, sign_lstm_best.h5
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

    # Candidates for the label/“actions” file: sign_actions_single.pkl and 
    # sign_actions.pkl for single and multiple gestures respectively for 
    # single gestures only
    ACTIONS_PATHS = [
        os.path.join(MODEL_DIR, 'sign_actions_single.pkl'),
        os.path.join(MODEL_DIR, 'sign_actions.pkl'),
    ]

    model = None
    actions = None
    sequence_length = 30  # default; will be overwritten from model

    # Loading the model and the gesture labels
    # Load trained LSTM model (TensorFlow/Keras format)
    # Reference: TensorFlow/Keras model loading (Abadi et al., 2016; Chollet, 2015)
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = load_model(model_path)
            # Input shape is (batch, T, F), so index 1 is sequence length T
            # T = temporal sequence length, F = feature dimensions
            sequence_length = model.input_shape[1]
            print(f"Model loaded. Sequence length: {sequence_length}")
            break

    if model is None:
        print("No model found. Train and save a model under MODEL_DIR before running this script.")
        return

    # Loading the gesture labels from the file
    for actions_path in ACTIONS_PATHS:
        if os.path.exists(actions_path):
            with open(actions_path, 'rb') as f:
                actions_data = pickle.load(f)

            # Support dict, list, or other array-like formats
            if isinstance(actions_data, dict):
                actions = np.array([actions_data[k] for k in sorted(actions_data.keys())])
            elif isinstance(actions_data, list):
                actions = np.array(actions_data)
            else:
                actions = np.array(actions_data)

            print(f"Loaded actions: {actions}")
            break

    if actions is None:
        print("Actions file not found in MODEL_DIR (e.g., sign_actions.pkl).")
        return

    # Initializing the webcam and setting the frame width and height
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe Holistic for pose, face, and hand detection
    # Reference: MediaPipe Holistic (Lugaresi et al., 2019)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Rolling state for inference: sequence = [] - sequence = [] is a function that maintains a rolling window of the last T frames of keypoints    
    sequence = []           # last T keypoint vectors: sequence = [] is a function that maintains a rolling window of the last T frames of keypoints
    prediction_buffer = []  # last N predicted indices: prediction_buffer = [] is a function that maintains a rolling window of the last N predicted indices
    confidence_buffer = []  # last N max confidences: confidence_buffer = [] is a function that maintains a rolling window of the last N max confidences
    current_detection = None # current detection: current_detection = None is a function that maintains the current detection
    current_confidence = 0.0 # current confidence: current_confidence = 0.0 is a function that maintains the current confidence

    # Email + history bookkeeping
    sign_last_email_time = {}  # sign -> last email timestamp
    sign_text_buffer = []      # "[timestamp] SIGN_NAME": "[timestamp] SIGN_NAME" is a function that maintains the sign text buffer

    frame_count = 0

    print("\nStarting real-time detection...")
    print("Press 'q' in the OpenCV window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirroring the frame so it behaves like a mirror for the user: frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 1)

        # Running MediaPipe Holistic and rendering landmarks: image, results = mediapipe_detection(frame, holistic)
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Flattening landmarks to a feature vector: keypoints = extract_keypoints(results)
        keypoints = extract_keypoints(results)

        # True if at least one hand is in the frame: hands_detected = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)
        hands_detected = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )

        frame_count += 1

        if hands_detected:
            # Updating the temporal sequence window: sequence.append(keypoints)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
        else:
            # Resetting the temporal state when hands disappear: sequence = []
            sequence = []
            prediction_buffer = []
            confidence_buffer = []
            current_detection = None

        # Running the model only every PREDICTION_INTERVAL frames: if frame_count % PREDICTION_INTERVAL == 0:       
        if frame_count % PREDICTION_INTERVAL == 0:
            # Checking if the sequence length is equal to the sequence length and hands are detected: if len(sequence) == sequence_length and hands_detected:
            if len(sequence) == sequence_length and hands_detected:
                # Forward pass through LSTM model: (1, T, F), e.g., (1, 30, 258)
                # LSTM processes temporal sequences for gesture classification
                # Reference: LSTM for sequence modeling (Hochreiter & Schmidhuber, 1997)
                probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_idx = int(np.argmax(probs))
                predicted_action = actions[predicted_idx]
                confidence = float(probs[predicted_idx])

                # Updating smoothing buffers: prediction_buffer.append(predicted_idx)
                prediction_buffer.append(predicted_idx)
                confidence_buffer.append(confidence)
                # Keeping only the last SMOOTHING_WINDOW predictions: prediction_buffer = prediction_buffer[-SMOOTHING_WINDOW:]
                prediction_buffer = prediction_buffer[-SMOOTHING_WINDOW:]
                confidence_buffer = confidence_buffer[-SMOOTHING_WINDOW:]

                # Temporal smoothing: require last N predictions to agree
                # This reduces prediction noise and improves stability
                # Reference: Temporal smoothing in sequence classification (Graves, 2012)
                if len(prediction_buffer) >= SMOOTHING_WINDOW:
                    unique_preds = np.unique(prediction_buffer)
                    if len(unique_preds) == 1:
                        avg_conf = float(np.mean(confidence_buffer))
                        if avg_conf > THRESHOLD:
                            current_detection = predicted_action
                            current_confidence = avg_conf

                            # Per-sign email cooldown
                            now = time.time()
                            last_time = sign_last_email_time.get(current_detection, 0.0)

                            if now - last_time > SIGN_EMAIL_COOLDOWN:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                sign_text_buffer.append(f"[{timestamp}] {current_detection}")

                                # Treat some labels as “emergency” to adjust email tone
                                emergency_signs = [
                                    'emergency', 'help', 'pain', 'hospital',
                                    'ambulance', 'fire', 'heart_attack'
                                ]
                                # Checking if the current detection is an emergency sign: if current_detection.lower() in emergency_signs:
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
                            current_detection = None
                    else:
                        current_detection = None
                else:
                    current_detection = None

        # Visual overlays 

        # Drawing the main detection banner at the top: if current_detection:
        if current_detection:
            label_text = current_detection.upper()
            font_scale = 1.5
            thickness = 3

            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            x = (image.shape[1] - text_w) // 2
            y = 50

            # Drawing the background rectangle: cv2.rectangle(image, (x - 10, y - text_h - 10), (x + text_w + 10, y + baseline + 10), (0, 255, 0), thickness=-1)
            cv2.rectangle(
                image,
                (x - 10, y - text_h - 10),
                (x + text_w + 10, y + baseline + 10),
                (0, 255, 0),
                thickness=-1
            )

            # Drawing the label text: cv2.putText(image, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(
                image,
                label_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

            # Drawing the confidence text: cv2.putText(image, conf_str, (x + text_w // 2 - 20, y + text_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            conf_str = f"{current_confidence * 100:.0f}%"
            cv2.putText(
                image,
                conf_str,
                (x + text_w // 2 - 20, y + text_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Short history of recent detections near the bottom: if sign_text_buffer:
        if sign_text_buffer:
            recent = [s.split("] ")[1] for s in sign_text_buffer[-3:]]
            history_text = "Detected Signs: " + ", ".join(recent)
            cv2.putText(
                image,
                history_text,
                (10, image.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # Sequence buffer status (helps debug whether the window fills correctly): status = f"Buffer: {len(sequence)}/{sequence_length}"
        status = f"Buffer: {len(sequence)}/{sequence_length}"
        cv2.putText(
            image,
            status,
            (10, image.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("Real-Time Sign Detection", image)

        # Pressing 'q' to exit the loop: if cv2.waitKey(10) & 0xFF == ord('q'): break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleaning up resources: releasing the webcam and closing the windows: cap.release() and cv2.destroyAllWindows() and holistic.close()
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
