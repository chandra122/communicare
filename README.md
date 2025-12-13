# Communicare: Sign Language Recognition & Fall Detection System

[![GitHub stars](https://img.shields.io/github/stars/chandra122/communicare?style=social)](https://github.com/chandra122/communicare)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A computer vision and machine learning system for real-time sign language translation and automatic fall detection.

---

## Table of Contents

- [What Is Communicare?](#what-is-communicare)
- [Why This Matters](#why-this-matters)
- [How It Works](#how-it-works)
  - [Sign Language Recognition](#sign-language-recognition-understanding-motion-through-time)
  - [Fall Detection](#fall-detection-sequence-based-learning-for-accurate-detection)
- [What I Achieved](#what-i-achieved)
- [What's Next](#whats-next)
- [Get Started](#get-started)
- [Built With](#built-with)
- [References](#references)
- [License](#license)
- [Contact & Contribute](#contact--contribute)

---

# What Is Communicare?

Imagine being unable to call for help. Imagine falling and having no one nearby to notice.

These scenarios happen every day, and I wanted to do something about it. That's why I built Communicare, a system that uses computer vision and machine learning to address two critical problems that affect millions of people worldwide.

**Sign Language Recognition** - Converts Sign Language gestures into text in real-time, helping deaf and hard-of-hearing individuals communicate with anyone, anywhere, without needing an interpreter. The system watches your hands through a webcam and instantly translates your gestures into readable text.

**Fall Detection** - Automatically alerts emergency contacts when someone falls, providing peace of mind for elderly individuals and their families. The system continuously monitors body movements and sends immediate email alerts when a fall is detected, ensuring help arrives quickly even when no one is physically present.

What makes Communicare special is that real-time detection runs entirely locally on your computer. There's no cloud processing during detection, no data being sent to external servers, just your webcam, your computer, and complete privacy. Model training happens on Google Colab using data you upload to Google Drive, but once trained, the models run completely offline on your local machine. I designed it this way because when it comes to health and communication, people deserve systems they can trust.


<img src="https://i.imgur.com/kk7Eqki.gif" width="300" height="200" alt="System detecting Sign example HELP sign and triggering emergency alert">

*System detecting Sign example: "HELP" sign and triggering emergency alert*

---

# Why This Matters

Over 430 million people worldwide have hearing disabilities [[WHO]](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss). Falls are the leading cause of injury deaths for older people [[CDC]](https://www.cdc.gov/falls/data-research/index.html). 

These aren't just statistics, they represent real people facing daily challenges. For the deaf and hard-of-hearing community, every conversation can become a barrier when others don't understand sign language. Hiring interpreters is expensive and not always available, leaving many feeling isolated in their own communities. For elderly individuals living alone, a simple fall can turn into a life-threatening situation if help doesn't arrive in time. Traditional fall detection systems often require wearable devices that many people find uncomfortable or forget to wear.

Because of these reasons, I created Communicare. 


---

# How It Works

## Sign Language Recognition: Understanding Motion Through Sequences

Hands tell a story through movement, not just positions. The sign for "Pain" and "Help" might look similar in a photo, but the motion is completely different. This is why I analyze sequences instead of individual frames.

### **Step-by-Step Process**

**1. Video Capture**  

The First step is to capture the webcam feed video at 30 frames per second. Each frame is just a snapshot, but sign language happens across time.

**2. Landmark Extraction**  

For every frame, MediaPipe Holistic detects and tracks:

- 33 body pose landmarks (shoulders, elbows, wrists, hips, knees, ankles)

- 21 landmarks per hand (42 total - tracking each finger joint and palm)

- 468 facial landmarks (for expressions that add context to signs)

These landmarks are converted into x, y, z coordinates. Instead of processing millions of raw pixels, I now have a clean numerical representation of body position.

**3. Sequence Formation**  

Here's the crucial part. I maintain a rolling window of 30 consecutive frames. Think of it like this:

- Frame 1-30: First sequence analyzed

- Frame 2-31: Second sequence analyzed  

- Frame 3-32: Third sequence analyzed

This sliding window continuously captures 1 second of motion, updating with each new frame.

**4. LSTM Processing**  

The 30-frame sequence feeds into a Long Short-Term Memory neural network. LSTM is perfect for this because it has memory, and it understands temporal patterns. When someone signs "HELP":

- Frames 1-10: Hands starting position detected

- Frames 11-20: Motion trajectory captured (one hand lifts the other)

- Frames 21-30: Final position and speed of movement recorded

The LSTM compares this pattern against learned sequences from training data. I trained it on 42 sequences per sign across 16 signs, totaling over 1,000+ sequences (42 sequences × 30 frames = 1,260 frames per gesture).

**5. Classification & Output**  

The network outputs a probability distribution across all known signs. If a sign scores above the confidence threshold (configurable, default 90%), the system displays the text immediately.

**Why 30 frames?**  

Too few frames miss the complete motion. Too many frames create lag and include irrelevant movements. Testing showed that 30 frames (1 second) captures most sign language signs completely while maintaining real-time performance.

<img src="https://i.imgur.com/cEnSTx8.gif" width="300" height="200" alt="Real-time gesture analysis and text conversion">

*Real-time gesture analysis and text conversion*

## Fall Detection: Sequence-Based Detection for Safety

Falls happen quickly, but they also have a distinct pattern over time. Just like sign language, I analyze sequences of frames to understand the complete motion, from standing to falling to impact. This temporal approach is more accurate than analyzing single frames.

### **Step-by-Step Process**

**1. Real-Time Pose Detection**  

YOLOv8-pose processes each frame as it arrives, extracting 17 body keypoints:

- Head, shoulders, elbows, wrists

- Hips, knees, ankles

- All keypoints include x, y coordinates and confidence scores

YOLOv8 is optimized for speed, it can process 30 frames per second even on a laptop CPU.

**2. Sequence Formation**  

Similar to sign language recognition, I maintain a rolling window of 30 consecutive frames. This captures the complete fall motion:

- Frames 1-10: Person standing or moving normally

- Frames 11-20: Rapid descent begins (if a fall occurs)

- Frames 21-30: Impact and post-fall position

Each frame contributes 51 features (17 keypoints × 3 values: x, y, confidence), creating a sequence of 30 frames × 51 features for the LSTM model.

**3. LSTM Processing**  

The 30-frame sequence feeds into a Long Short-Term Memory neural network trained on fall and normal activity data. The LSTM learns to recognize the temporal patterns that distinguish falls from other activities:

- Rapid downward movement followed by horizontal positioning

- Sudden changes in body orientation

- Post-impact stillness patterns

The model outputs a probability: "fall" or "normal" activity.

**4. Temporal Smoothing**  

To prevent false alarms from single-frame glitches, I apply temporal smoothing:

- The model runs every 3 frames (not every frame) to balance speed and accuracy

- Confidence scores from the last 5 predictions are averaged

- Only predictions above 70% confidence are displayed

**5. Alert Sequence**  

Once a fall is detected with high confidence:

- Timestamp captured

- Screenshot saved

- Email composed with incident details

- Email sent to all emergency contacts (typically arrives within 5-8 seconds)

**Why sequence-based detection?**  

Single-frame analysis misses the context of motion. Someone bending down looks similar to someone falling in one frame, but the sequence tells the story. The LSTM model learned from training data to recognize the complete pattern of a fall—the acceleration, impact, and aftermath, resulting in 95% accuracy with minimal false positives.

<img src="https://i.imgur.com/uUTWZsU.gif" width="300" height="200" alt="Emergency alert triggered after fall detection">

*Emergency alert triggered after fall detection*

---

# What I Achieved

**Sign Language Performance:**

- 98.64% test accuracy across 16 ASL signs:
  - ambulance, cold, cough, dizzy, doctor, emergency, fever, fire, help, hot, hungry, pain, sick, sneeze, spread, thirsty

- Instant recognition (less than 100ms after gesture)

- Works with different signing speeds and styles

<img src="https://i.imgur.com/wAbxaF0.png" width="300" height="300" alt="Model training accuracy and loss curves">

*Model training accuracy and loss curves*

<img src="https://i.imgur.com/giyBl9C.png" width="300" height="300" alt="Sign language recognition confusion matrix">

*Sign language recognition confusion matrix*

**Fall Detection Performance:**

- 95.00% test accuracy

- Distinguishes between falls and normal activities with minimal false positives

- Alerts sent within 5-8 seconds

<img src="https://i.imgur.com/WQVRHuJ.png" width="300" height="300" alt="Fall detection performance metrics">

*Fall detection performance metrics*

<img src="https://i.imgur.com/r0R3Khi.png" width="300" height="300" alt="Fall detection model training curves">

*Fall detection model training curves*

**System Specs:**

- Model training runs on GPU (Google Colab) for faster training times

- Real-time detection runs on regular laptops (no GPU needed)

- 20-30 frames per second

- Works completely offline (internet only for email alerts)

- Both systems run simultaneously

---

# What's Next

Looking ahead, there are two main directions where I want to take Communicare.

First is microcontroller integration, so the system can run as a low-power, battery-operated, portable device that does not need a full computer, something that could be placed in different rooms or even carried between locations.

Second is IoT and smart-home integration, where Communicare could push alerts over Wi-Fi to phones, watches, and smart speakers, and support multiple cameras around a home for broader coverage, especially in elder-care settings.

---

# Get Started

Want to try it? Check out the **[Installation Guide](https://github.com/chandra122/communicare/blob/master/installation_guide.md)** for setup instructions.

For a detailed overview of the project workflow, see the **[Project Flow](https://github.com/chandra122/communicare/blob/master/project_flow.md)** documentation.

---

# Built With

**Computer Vision & Detection:**

- MediaPipe Holistic - Hand, face, and pose tracking

- YOLOv8 - Fast pose estimation for fall detection

- OpenCV - Video processing

**Machine Learning:**

- TensorFlow/Keras - LSTM model for gesture recognition

- NumPy & SciPy - Data processing

**References:**

- [Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model](https://www.youtube.com/watch?v=doDUihpj6ro&t=462s)

- [Emergency Signs](https://www.youtube.com/watch?v=GW-9-SujCqQ&t=8s)

- [AI Pose Estimation with Python and MediaPipe | Plus AI Gym Tracker Project](https://www.youtube.com/watch?v=Hk-Os9QqkZQ)

- [MediaPipe](https://mediapipe.dev/) by Google Research

- [YOLOv8](https://docs.ultralytics.com/) by Ultralytics

---

## License

GNU General Public License v3.0 - See LICENSE file for details

---

## Contact & Contribute

Found this useful? Have ideas to make it better? 

**[GitHub Repository](https://github.com/chandra122/communicare)**

Star this repo - Report issues - Suggest features.

---

*Built for accessibility and safety*
