# Bulls-Eye --> Real-Time Person Tracking & Re-Identification Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning pipeline built for hackathon submission that detects, tracks, and re-identifies individuals in a video stream, assigning a persistent ID even when they temporarily disappear from view.

---

# For the Jury \^_^
## Setup and Installation

Cheatcode: run this on google colab instead: [Colab link!](https://colab.research.google.com/drive/1nq1NOA40PXEB9Gvg3OEM3ucormTHj2lX?usp=sharing)

Follow these steps to get the project running.


**1. Clone the repository:**
```bash
git clone https://github.com/RAP-Team-AURA/bulls-eye.git
cd bulls-eye
pip install -r requirements.txt
python RAP3.py
```

---

## The Problem

In standard video surveillance or analytics, simple object trackers assign a new ID to a person every time they re-enter the frame after being occluded or leaving. This makes it impossible to perform long-term analysis, like tracking a customer's journey through a store or monitoring a person of interest in a security feed.

## The Solution

This project solves the re-identification problem by creating a "digital fingerprint" for each person using their face. When a person is detected, their face is encoded. If they disappear and reappear later, the system recognizes their face and re-assigns their original ID, ensuring persistent tracking.

---

## ⭐ Key Features

* **Real-time Person Detection:** Uses the powerful YOLOv5 model to detect people with high accuracy.
* **Multi-Person Tracking:** Implements a robust tracker using Intersection over Union (IoU) and centroid distance to follow multiple people simultaneously.
* **Face-Based Re-Identification:** The core feature. It extracts facial embeddings to recognize individuals who have previously been seen.
* **Persistent ID Assignment:** Ensures a person keeps the same ID throughout the video, even after leaving and re-entering the frame.
* **Comprehensive Data Export:** Automatically saves processed frames, cropped face images for each person, and a detailed JSON log of all detection events.

---

## 🛠️ Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core programming language. |
| **PyTorch** | For running the YOLOv5 model. |
| **YOLOv5** | State-of-the-art object detection model for identifying people. |
| **OpenCV** | For video processing, face detection (Haar Cascades), and visualization. |
| **`face_recognition` (dlib)** | For extracting 128-d facial embeddings for re-identification. |
| **NumPy** | For efficient numerical operations. |

---

## ⚙️ How It Works: The Process Flow

The pipeline processes each video frame through a series of steps to achieve its goal:

**1. Frame Ingestion**
* The system captures a frame from the video source (webcam or file).

**2. Person Detection (YOLOv5)**
* The entire frame is passed to a pre-trained YOLOv5 model.
* The model returns the bounding boxes for all detected objects with the class 'person'.

**3. Face Detection & Encoding**
* For each detected person, the system runs a Haar Cascade classifier within their bounding box to find a face.
* If a face is found, the `face_recognition` library converts it into a unique 128-point numerical vector (a "face encoding").

**4. Re-Identification Logic**
* The new face encoding is compared against a database of known encodings from previously seen individuals.
    > * **Is it a Match?** If the similarity is above a set threshold (e.g., 60%), the person is recognized. Their existing, persistent ID is retrieved.
    > * **Is it a New Person?** If no match is found, this individual is considered new. Their face encoding is added to the database, ready to be assigned a new ID in the next step.

**5. Tracking & State Update**
* The system uses an IoU and distance-based tracker to associate the current detections with the tracks from the previous frame.
    > * **Update:** A matched person's location is updated.
    > * **Register:** An unmatched new person is assigned a new ID. Their face encoding (if available) is now linked to this ID.
    > * **Deregister:** If a tracked person is missing for too many frames, their track is removed until they are re-identified.

**6. 🎨 Visualization & Export**
* The final, processed frame is displayed with bounding boxes, persistent IDs, and tracking trails.
* Periodically, the system saves the frame, any detected face crops, and updates the `detection_log.json` file.

---

