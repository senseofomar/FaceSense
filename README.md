# 🎭 FaceSense – End-to-End Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![MySQL](https://img.shields.io/badge/Database-MySQL-blue)

**FaceSense** is a full-stack computer vision application designed to analyze, log, and visualize human facial expressions in real-time. It goes beyond simple detection by integrating a transactional database logger, a "Sci-Fi" style HUD, and a forensic analytics dashboard.

---

## 💡 The Use Case
Imagine installing this system in a **Movie Theater**. Instead of relying on written reviews, studios could analyze audience reactions second-by-second—identifying exactly when the crowd laughed, screamed, or got bored. FaceSense turns human reaction into actionable data.

---

## 🚀 Key Features

### 1. Real-Time Inference Engine (`live.py`)
* **Deep Learning:** Uses **DeepFace** (TensorFlow/Keras) for emotion classification.
* **Custom Stabilization:** Implements a **Rolling Vote Buffer (Deque)** to eliminate label flickering and jitter.
* **"Highlander" Logic:** Custom algorithm to filter "Ghost Faces" (false positives) by calculating bounding box area and strictly tracking the primary subject.
* **Sci-Fi HUD:** Custom OpenCV drawing logic featuring a dynamic confidence bar, scanning laser animation, and real-time FPS telemetry.

### 2. Intelligent Data Logging
* **Concurrency Handling:** Solved "Race Conditions" between the high-speed video writer (30 FPS) and the database logger using a non-blocking architecture.
* **Session Management:** Prevents "Data Pollution" by only logging telemetry when an active session is triggered via the dashboard.

### 3. Analytics Dashboard (`facesense_dashboard.py`)
* **Interactive UI:** Built with **Streamlit** for real-time data visualization.
* **Forensics:** Review past sessions with breakdowns of "Dominant Emotions" and confidence trends.
* **Live Monitor:** Watch the webcam feed remotely via the dashboard.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **AI Model:** DeepFace (FER2013 weights)
* **Database:** MySQL (Connector/Python)
* **Visualization:** Streamlit, Altair, Pandas
* **Data Processing:** NumPy

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/senseofomar/FaceSense.git](https://github.com/senseofomar/FaceSense.git)
cd FaceSense
2. Set up Virtual Environment
Bash

python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Database Setup (MySQL)
Create a database named facesense_db and run the following SQL commands:

SQL

CREATE TABLE sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_name VARCHAR(255),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE emotion_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_ref_id INT,
    expression VARCHAR(50),
    confidence FLOAT,
    x1 INT, y1 INT, x2 INT, y2 INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_ref_id) REFERENCES sessions(id)
);
5. Configure Database Connection
Update the facesense/storage/db.py file with your MySQL credentials:

Python

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="your_user",
        password="your_password",
        database="facesense_db"
    )
🖥️ Usage
Step 1: Start the Dashboard
This controls the session recording.

Bash

streamlit run app/facesense_dashboard.py
Step 2: Start the Camera Feed
This runs the AI inference.

Bash

python facesense/apps/live.py
Press 'q' to quit the camera.

Status Indicator: Watch the "LOGS" text in the top-left. It will switch from IDLE (Gray) to ACTIVE (Red) when you start a session on the dashboard.

🧠 Engineering Challenges Solved
Ghost Face Detection: Haar Cascades often detect shadows on the neck as faces. I implemented a size-filtering logic that dynamically assesses all detected rectangles and discards non-primary faces.

File Locking (Race Condition): The dashboard tries to read the latest frame while the camera writes it. I implemented a Retry-with-Fallback mechanism to ensure the video feed remains smooth (30 FPS) without crashing due to Windows file access errors.

🤝 Contributing
Open to feedback! If you have ideas for other use cases (retail analysis, online education monitoring, etc.), feel free to open an issue or pull request.

Author: senseofomar