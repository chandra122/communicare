"""
Sign Language Recognition - Data Collection Script:

This script collects temporal sequence data for sign language recognition training.
It uses MediaPipe Holistic to extract pose, face, and hand landmarks from webcam
video and saves them as sequences for LSTM model training.

"""

# Importing all the libraries I need for data collection
import cv2  # OpenCV: Computer vision library (Bradski, 2000)
import os
import mediapipe as mp  # MediaPipe: Holistic pose estimation (Lugaresi et al., 2019)
import numpy as np  # NumPy: Numerical computing (Harris et al., 2020)
from datetime import datetime


# Feature selection configuration (face on/off)
USE_FACE_FEATURES = input("Do you want to use face features? (y/n): ").strip().lower()
USE_FACE_FEATURES = (USE_FACE_FEATURES == "y")

# If True:
#   Pose  (132) + Face (1404) + Left Hand (63) + Right Hand (63) = 1662 features
# If False:
#   Pose  (132) + Face (1404 zeros) + Left Hand (63) + Right Hand (63) = 1662 features
# In both cases, I always end up with 1662 features per frame so the model input is consistent.


# Initializing MediaPipe Holistic and drawing utilities
# Reference: MediaPipe Holistic (Lugaresi et al., 2019)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# I keep one Holistic instance alive for the whole session
# MediaPipe Holistic provides unified pose, face, and hand detection
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Keypoint extraction

def extract_keypoints(results):
    """
    Building a flat feature vector from MediaPipe Holistic results.
    
    This function extracts pose, face, and hand landmarks from MediaPipe Holistic
    detection results and concatenates them into a single feature vector suitable
    for temporal sequence modeling.
    
    Reference: MediaPipe Holistic provides 33 pose landmarks, 468 face landmarks,
    and 21 landmarks per hand (Lugaresi et al., 2019; Bazarevsky et al., 2020).

    Input:
        results: output of holistic.process(frame_rgb) for a single frame.

    Output:
        keypoints: 1D NumPy array of length 1662:
            Pose:  33 * 4  = 132
            Face: 468 * 3 = 1404 (values or zeros, depending on USE_FACE_FEATURES)
            Left hand: 21 * 3 = 63
            Right hand:21 * 3 = 63
    """
    # Pose landmarks (x, y, z, visibility)
    pose = np.array(
        [[res.x, res.y, res.z, res.visibility]
         for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # Face landmarks (x, y, z); I always reserve space, but values depend on USE_FACE_FEATURES
    if USE_FACE_FEATURES:
        # Use face landmarks if present; otherwise, zeros
        face = np.array(
            [[res.x, res.y, res.z]
             for res in results.face_landmarks.landmark]
        ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    else:
        # Force face block to zeros (even if MediaPipe detects it)
        face = np.zeros(468 * 3)

    # Left hand landmarks (x, y, z)
    lh = np.array(
        [[res.x, res.y, res.z]
         for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Right hand landmarks (x, y, z)
    rh = np.array(
        [[res.x, res.y, res.z]
         for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # Final feature vector is always 1662 values
    return np.concatenate([pose, face, lh, rh])


# Camera initialization
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Base sequence length for a simple, single sign
sequence_length = 30  # 30 frames ≈ 1 second at 30 FPS

# Where I store my MediaPipe sequences
data_dir = "MP_Data"

# Class selection & mode configuration
print("DATA COLLECTION FOR HAND SIGN RECOGNITION")

print("\nFeature Configuration:")
print(f"  USE_FACE_FEATURES: {USE_FACE_FEATURES}")
if USE_FACE_FEATURES:
    print("  Features: Pose(132) + Face(1404) + Left(63) + Right(63) = 1662")
    print("  Face values are real when detected, zeros if not.")
else:
    print("  Features: Pose(132) + Face(1404 zeros) + Left(63) + Right(63) = 1662")
    print("  Face block is always zeros (hand-focused data).")
print("  Total features per frame: 1662")

print("\nWhich sign class do you want to collect?")
print("Examples: help, water, emergency, pain, thank_you, etc.")
current_class = input("Enter sign class name: ").strip().lower()

if not current_class:
    print("Error: No class name provided. Exiting.")
    exit(1)

# Single gesture mode: 30 frames per sequence
sequence_length = 30  # 30 frames ≈ 1 second at 30 FPS
print("\nSingle gesture mode: using 30 frames per sequence.")

print(f"\nCollecting data for class: '{current_class}'")
print(f"Sequences will be saved to: {data_dir}/{current_class}/")
print(f"Each sequence contains {sequence_length} frames of keypoints.")

# Directory setup and sequence numbering
class_dir = os.path.join(data_dir, current_class)
os.makedirs(class_dir, exist_ok=True)

# Find existing sequence ids so I can continue from the last one
existing_sequences = []
if os.path.exists(class_dir):
    for item in os.listdir(class_dir):
        item_path = os.path.join(class_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            try:
                existing_sequences.append(int(item))
            except ValueError:
                pass

if existing_sequences:
    sequence_counter = max(existing_sequences) + 1
    print(f"\nFound {len(existing_sequences)} existing sequences (0 to {max(existing_sequences)})")
    print(f"Continuing from sequence {sequence_counter}")
else:
    sequence_counter = 0
    print("\nStarting fresh collection (no existing sequences).")

# These variables track the recording state
recording = False           # True when we're currently recording a sequence
current_sequence = []       # Stores keypoints for the sequence we're recording
frame_count = 0             # Counts frames in the current sequence

# Burst mode: automatically records sequences without pressing SPACE each time
burst_mode = False          # When True, sequences are recorded automatically
burst_interval = 2.0        # Wait 2 seconds between automatic recordings
last_burst_time = datetime.now().timestamp()  # Track when we last recorded

# Main loop: capture frames and save sequences
print("\nStarting camera feed for data collection...")
print(f"  Press SPACE to record a sequence of {sequence_length} frames (manual mode).")
print("  Press 'b' to toggle burst mode (auto-record sequences).")
print("  Press 'q' to quit.")

while True:
    success, frame = capture.read()
    if not success:
        print("Error: Could not read frame from camera.")
        break

    # Mirror the frame for a more natural feel
    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # Pose skeleton (includes some head points)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Face mesh (468 points) - only drawn if I chose to use face features
    if USE_FACE_FEATURES and results.face_landmarks:
        mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    # Hands
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Manual recording mode: user presses SPACE to start recording
    if recording:
        # Extract keypoints from this frame and add to our sequence
        keypoints = extract_keypoints(results)
        current_sequence.append(keypoints)
        frame_count += 1

        # Once we've collected 30 frames, save the entire sequence to disk
        # Each sequence is saved in its own folder with numbered .npy files
        if frame_count >= sequence_length:
            sequence_dir = os.path.join(data_dir, current_class, str(sequence_counter))
            os.makedirs(sequence_dir, exist_ok=True)

            for i, kp in enumerate(current_sequence):
                npy_path = os.path.join(sequence_dir, f"{i}.npy")
                np.save(npy_path, kp)

            sequence_counter += 1
            print(f"Sequence {sequence_counter} saved to {sequence_dir}/")

            # Reset state for the next manual sequence
            recording = False
            current_sequence = []
            frame_count = 0

            # For burst mode, I also update the last burst time
            if burst_mode:
                last_burst_time = datetime.now().timestamp()

    # Burst mode: automatically records sequences at regular intervals
    # This is useful when you want to collect many sequences quickly
    if burst_mode and not recording:
        current_time = datetime.now().timestamp()
        # Check if enough time has passed since last recording
        if current_time - last_burst_time >= burst_interval:
            # Only start recording if we can see hands or body
            # This prevents recording empty frames
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                recording = True
                current_sequence = []
                frame_count = 0
                print(f"Burst mode: started recording sequence {sequence_counter + 1}...")

    # On-screen status overlays
    cv2.putText(frame, f"Class: {current_class.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show face-feature configuration
    if USE_FACE_FEATURES:
        face_status = "FACE FEATURES: ON (1662 features)"
        face_color = (0, 255, 0)
    else:
        face_status = "FACE FEATURES: ZEROS (1662 features)"
        face_color = (0, 255, 255)
    cv2.putText(frame, face_status,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    # Show mode
    if burst_mode:
        mode_text = "BURST MODE: ON (auto-record)"
        mode_color = (0, 255, 0)
    else:
        mode_text = "MANUAL MODE: Press SPACE"
        mode_color = (0, 0, 255)
    cv2.putText(frame, mode_text,
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    if recording:
        cv2.putText(frame,
                    f"Recording... Frame {frame_count + 1}/{sequence_length}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Sequences: {sequence_counter}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if burst_mode:
            cv2.putText(frame, "Auto-recording...",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press SPACE to record",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Data Collection", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('b'):
        # Toggle burst mode
        burst_mode = not burst_mode
        if burst_mode:
            last_burst_time = datetime.now().timestamp()
            print("Burst mode ON - sequences will be recorded automatically.")
        else:
            print("Burst mode OFF - use SPACE for manual recording.")

    elif key == 32:  # SPACE
        if not recording and not burst_mode:
            # Starting a new manual recording
            recording = True
            current_sequence = []
            frame_count = 0
            print(f"Started recording sequence {sequence_counter + 1}...")

# Cleaning up resources

capture.release()
cv2.destroyAllWindows()
holistic.close()

print("\nCamera feed closed.")
print(f"Total sequences saved: {sequence_counter}")
print(f"Data saved under: {data_dir}/{current_class}/")


# REFERENCES
"""
Complete Reference List:

1. MediaPipe Holistic:
   Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines.
   arXiv preprint arXiv:1906.08172. https://arxiv.org/abs/1906.08172
   
   Bazarevsky, V., et al. (2020). BlazePose: On-device Real-time Body Pose tracking.
   arXiv preprint arXiv:2006.10204. https://arxiv.org/abs/2006.10204

2. OpenCV:
   Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
   https://opencv.org/

3. NumPy:
   Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.
   https://doi.org/10.1038/s41586-020-2649-2

4. Sign Language Data Collection:
   Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey.
   Expert Systems with Applications, 164, 113794.
   https://doi.org/10.1016/j.eswa.2020.113794

5. Temporal Sequence Data Collection:
   Graves, A. (2012). Supervised sequence labelling with recurrent neural networks.
   Springer Science & Business Media.

6. Hand Pose Estimation:
   Zhang, F., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking.
   arXiv preprint arXiv:2006.10214. https://arxiv.org/abs/2006.10214

7. Computer Vision for Sign Language:
   Ong, E. J., et al. (2012). Sign language recognition using sequential pattern trees.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
   (pp. 2200-2207).

8. Real-time Video Processing:
   Bradski, G., & Kaehler, A. (2008). Learning OpenCV: Computer vision with the OpenCV library.
   O'Reilly Media.
"""