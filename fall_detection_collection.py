"""
Fall Detection Data Collection Script

In this script, I collect sequence data for my fall detection model using
YOLOv8-pose. For each frame, I extract 51 features:
17 keypoints * 3 values (x, y, confidence).
"""

import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO


# Loading YOLOv8-POSE AND CAMERA

print("Loading YOLOv8-pose model...")
model = YOLO('yolov8n-pose.pt')
print("Model loaded successfully.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Configuration

SEQUENCE_LENGTH = 30        # frames per sequence (~1 second at 30 fps)
DATA_DIR = "Fall_Data"      # where I store all fall/normal/other sequences

# Names are just for my own reference; they're not used programmatically here
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Function: extract_keypoints

def extract_keypoints(results):
    """
    Turn YOLOv8-pose output into a flat feature vector for one frame.

    Input:
        results: list of YOLO Results objects from model(frame).
                 I use results[0] (first image in the batch).

    Output:
        features: np.ndarray of shape (51,)
            If a person is detected:
                17 keypoints × (x, y, confidence) = 51 values.
                The vector is [x1, y1, c1, x2, y2, c2, ...].
            If no person is detected:
                51 zeros.
    """
    if len(results[0].keypoints.data) == 0:
        # No person visible in this frame
        return np.zeros(51, dtype=np.float32)

    # Take the first (most confident) person's keypoints
    keypoints = results[0].keypoints.data[0].cpu().numpy()

    # Flatten (17, 3) -> (51,)
    features = keypoints.flatten().astype(np.float32)
    return features

# Initializing setup: choosing class and mode

print("FALL DETECTION DATA COLLECTION")
print("\nActivity classes I usually collect:")
print("  - normal: standing, walking, sitting, etc.")
print("  - fall:   forward/backward/side falls")
print("  - other:  bending, lying down, or anything else")
print("\nWhich activity class do you want to collect?")

current_class = input("Enter class name (normal/fall/other): ").strip().lower()
if not current_class:
    print("Error: No class name provided. Exiting.")
    raise SystemExit(1)

# Creating base directory for this class
class_dir = os.path.join(DATA_DIR, current_class)
os.makedirs(class_dir, exist_ok=True)

# Figuring out where to start sequence numbering
existing_sequences = [
    int(f) for f in os.listdir(class_dir)
    if os.path.isdir(os.path.join(class_dir, f)) and f.isdigit()
]
sequence_num = max(existing_sequences) + 1 if existing_sequences else 0

print(f"\nClass: {current_class}")
if existing_sequences:
    print(f"Existing sequences: {len(existing_sequences)} (0 to {max(existing_sequences)})")
    print(f"Next sequence: {sequence_num}")
else:
    print("No existing sequences found.")
    print(f"Starting at sequence: {sequence_num}")
print(f"Sequence length: {SEQUENCE_LENGTH} frames")
print("Features/frame: 51 (17 keypoints * 3 values)")

print("\nCollection modes:")
print("  1. Manual mode     - I arm the system with SPACE, it auto-counts down, records one sequence.")
print("  2. Continuous mode - It continuously watches and auto-extracts sequences when a person is present.")
mode = input("Select mode (1/2): ").strip()

continuous_mode = (mode == "2")

if continuous_mode:
    frame_buffer = []
    max_buffer_size = 60
    auto_extract_threshold = SEQUENCE_LENGTH
    last_extract_time = datetime.now().timestamp()
    continuous_extract_interval = 2.0  # seconds between automatic extractions
    min_person_frames = 5              # need some stability before buffering
    person_detected_count = 0

    print("\nContinuous mode ON")
    print(f"  - Needs person for at least {min_person_frames} frames before buffering")
    print(f"  - Waits at least {continuous_extract_interval} seconds between saves")
    print(f"  - Buffer capacity: {max_buffer_size} frames")
else:
    print("\nManual mode ON")
    print("  - Press SPACE to arm the system.")
    print("  - Once armed, step into the frame; countdown (3, 2, 1) starts automatically.")
    print("  - It records exactly one sequence of frames, then stops.")
    print("  - Press SPACE again while armed to cancel.")

print("\nPress 'q' to quit.")

# State for both modes

sequence_buffer = []           # used in manual mode for one sequence
recording = False
frame_count = 0

# Manual mode: countdown + arming
countdown_active = False
countdown_start_time = None
countdown_duration = 3.0       # seconds
armed = False                  # manual "armed" flag

# Main loop

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    frame = cv2.flip(frame, 1)  # mirror effect for the user
    frame_count += 1

    # Run YOLOv8-pose on this frame
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Extract features and number of people in this frame
    num_people = len(results[0].keypoints.data)
    keypoints = extract_keypoints(results)

    # Top status overlays

    mode_text = "CONTINUOUS MODE" if continuous_mode else "MANUAL MODE"
    mode_color = (255, 255, 0) if continuous_mode else (0, 255, 255)
    cv2.putText(
        annotated_frame, mode_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2
    )

    info_text = f"Class: {current_class} | Sequence: {sequence_num}"
    cv2.putText(
        annotated_frame, info_text, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

    if not continuous_mode and countdown_active:
        cv2.putText(
            annotated_frame, "COUNTDOWN ACTIVE - watch for 3, 2, 1",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3
        )

    if not continuous_mode and armed:
        cv2.putText(
            annotated_frame, ">>> SYSTEM ARMED <<<",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3
        )

    # Continuous mode:
    if continuous_mode:
        # Keeping track of how many frames we've seen a person
        person_detected = num_people > 0
        if person_detected:
            person_detected_count += 1
        else:
            person_detected_count = 0

        # Only start buffering once a person has been present a little while
        if person_detected_count >= min_person_frames:
            frame_buffer.append(keypoints)

            # Limiting buffer size so it doesn't grow forever
            if len(frame_buffer) > max_buffer_size:
                frame_buffer.pop(0)

            # Decide whether to extract a sequence
            now = datetime.now().timestamp()
            time_since_last = now - last_extract_time

            if len(frame_buffer) >= auto_extract_threshold and time_since_last >= continuous_extract_interval:
                # Taking the last SEQUENCE_LENGTH frames as one sequence
                sequence_data = frame_buffer[-SEQUENCE_LENGTH:]

                seq_dir = os.path.join(class_dir, str(sequence_num))
                os.makedirs(seq_dir, exist_ok=True)

                for i, frame_data in enumerate(sequence_data):
                    np.save(os.path.join(seq_dir, f"{i}.npy"), frame_data)

                print(f"Sequence {sequence_num} saved ({len(sequence_data)} frames).")
                sequence_num += 1
                last_extract_time = now

                # Clearing buffer and resetting detection count
                frame_buffer = []
                person_detected_count = 0

        # Showing buffer and readiness status on screen
        buffer_text = f"Buffer: {len(frame_buffer)}/{max_buffer_size} | People: {num_people}"
        cv2.putText(
            annotated_frame, buffer_text, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

        if person_detected_count < min_person_frames:
            status_text = f"Waiting for person ({person_detected_count}/{min_person_frames})"
            color = (0, 165, 255)
        else:
            status_text = "Ready to extract"
            color = (0, 255, 0)

        cv2.putText(
            annotated_frame, status_text, (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Manual mode:
    else:
        # If armed but not yet counting down, wait for a person to appear
        if armed and not countdown_active and not recording:
            if num_people > 0:
                countdown_active = True
                countdown_start_time = datetime.now().timestamp()
                armed = False  # disarm as soon as countdown starts

                print(f"\n>>> PERSON DETECTED - COUNTDOWN FOR SEQ {sequence_num} <<<")
                print("Watch the video window: you'll see 3, 2, 1 before recording.")

        # Handle countdown
        if countdown_active:
            elapsed = datetime.now().timestamp() - countdown_start_time
            remaining = countdown_duration - elapsed

            if remaining <= 0:
                # Countdown done -> start recording sequence
                countdown_active = False
                recording = True
                sequence_buffer = []
                print(f"Recording sequence {sequence_num}...")

        elif recording:
            # Actually collecting frames while recording
            sequence_buffer.append(keypoints)
            progress = len(sequence_buffer)

            status_text = f"Recording: {progress}/{SEQUENCE_LENGTH}"
            cv2.putText(
                annotated_frame, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # If I have enough frames, saving the sequence
            if len(sequence_buffer) >= SEQUENCE_LENGTH:
                seq_dir = os.path.join(class_dir, str(sequence_num))
                os.makedirs(seq_dir, exist_ok=True)

                for i, frame_data in enumerate(sequence_buffer):
                    np.save(os.path.join(seq_dir, f"{i}.npy"), frame_data)

                print(f"Sequence {sequence_num} saved ({len(sequence_buffer)} frames).")
                sequence_num += 1
                sequence_buffer = []
                recording = False

        else:
            # Not armed, not recording, not counting down:
            if armed:
                status_text = "ARMED – step into frame; countdown will start automatically."
                cv2.putText(
                    annotated_frame, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3
                )
                if num_people == 0:
                    wait_text = "Waiting for person..."
                    cv2.putText(
                        annotated_frame, wait_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
            elif num_people > 0:
                status_text = "Press SPACE to arm the system, then get in position."
                cv2.putText(
                    annotated_frame, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            else:
                status_text = "No person detected - step into the frame."
                cv2.putText(
                    annotated_frame, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )

    # Person count at the bottom:
    person_text = f"People detected: {num_people}"
    cv2.putText(
        annotated_frame, person_text,
        (10, annotated_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Drawing countdown overlay last so it sits on top of everything:
    if not continuous_mode and countdown_active:
        elapsed = datetime.now().timestamp() - countdown_start_time
        remaining = countdown_duration - elapsed

        if remaining > 0:
            countdown_num = int(remaining) + 1
            countdown_text = str(countdown_num)

            h, w = annotated_frame.shape[:2]

            # Slightly darkening entire frame:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

            # White circle at center:
            cv2.circle(annotated_frame, (w // 2, h // 2), 150, (255, 255, 255), -1)
            cv2.circle(annotated_frame, (w // 2, h // 2), 150, (0, 0, 0), 5)

            # Big red countdown number:
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 8, 15)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(
                annotated_frame, countdown_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 15
            )

            # "Starting in:" label above:
            start_msg = "Starting in:"
            start_size = cv2.getTextSize(start_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            start_x = (w - start_size[0]) // 2
            cv2.putText(
                annotated_frame, start_msg,
                (start_x, text_y - text_size[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
            )

    # Showing the frame:
    cv2.imshow('Fall Detection Data Collection', annotated_frame)

    # Keeping the window on top during countdown so the user sees it:
    if not continuous_mode and countdown_active:
        cv2.setWindowProperty('Fall Detection Data Collection', cv2.WND_PROP_TOPMOST, 1)

    # Keyboard controls:    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not continuous_mode and not recording and not countdown_active:
        # SPACE toggles the armed state in manual mode when idle:
        if armed:
            armed = False
            print(">>> System disarmed <<<")
        else:
            armed = True
            print(f"\n>>> SYSTEM ARMED for sequence {sequence_num} <<<")
            print("Step out, then step back into the frame. Countdown will start automatically.")
            print("(Press SPACE again to cancel.)")

# Cleaning up:

cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")
print(f"Total sequences collected: {sequence_num}")
print(f"Data saved under: {class_dir}")
