import cv2

import numpy as np
import time

from collections import deque # this is to add buffer to emotion, stable feed

from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion

# Define colors (BGR format for OpenCV)
COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)

# We define this OUTSIDE the loop for efficiency
COLORS = {
    'angry': (0, 0, 255),    # Red
    'happy': (0, 255, 255),  # Yellow
    'sad': (255, 0, 0),      # Blue
    'neutral': (0, 255, 0),  # Green
    'surprise': (255, 255, 0), # Cyan
    'fear': (128, 0, 128),   # Purple
    'disgust': (0, 128, 0)   # Dark Green
}

def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def get_dominant_emotion(buffer):
    """Returns the most frequent emotion in the buffer (Stabilization)"""
    if not buffer:
        return "neutral"
    return max(set(buffer), key=buffer.count)


def draw_tech_ui(frame, x, y, w, h, color, scan_line_pos):
    """Draws a Sci-Fi / Iron Man style HUD around the face"""

    # 1. Corner Brackets (The "Targeting" look)
    # Top-Left
    cv2.line(frame, (x, y), (x + 20, y), color, 2)
    cv2.line(frame, (x, y), (x, y + 20), color, 2)
    # Top-Right
    cv2.line(frame, (x + w, y), (x + w - 20, y), color, 2)
    cv2.line(frame, (x + w, y), (x + w, y + 20), color, 2)
    # Bottom-Left
    cv2.line(frame, (x, y + h), (x + 20, y + h), color, 2)
    cv2.line(frame, (x, y + h), (x, y + h - 20), color, 2)
    # Bottom-Right
    cv2.line(frame, (x + w, y + h), (x + w - 20, y + h), color, 2)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - 20), color, 2)

    # 2. Scanning Laser Line (Moves up and down)
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), (0, 255, 0), 2)
    # Add a "glow" effect to the line
    overlay = frame.copy()
    cv2.line(overlay, (x, scan_y), (x + w, scan_y), (150, 255, 150), 4)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # 3. Tech Decorators (Dots and text)
    # Center Point
    cv2.circle(frame, (x + w // 2, y + h // 2), 2, COLOR_WHITE, -1)

    # Side Data Block
    cv2.putText(frame, f"ID: 0x{id(x) % 1000:03X}", (x + w + 5, y + 20),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    cv2.putText(frame, f"TRK: {int(scan_line_pos * 100)}%", (x + w + 5, y + 40),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

def main():
    cap = init_camera()
    frame_count = 0

    # --- STABILIZATION BUFFER ---
    # Stores the last 7 emotions to prevent flickering
    emotion_window = deque(maxlen=7)
    current_stable_emotion = "neutral"

    # Scanner animation variables
    scan_pos = 0.0
    scan_direction = 0.05

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Mirror
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # 2.The "Iron Man" Layer
        scan_pos += scan_direction
        if scan_pos >= 1.0 or scan_pos <= 0.0:
            scan_direction *= -1

        # 3. EMOTION LOGIC
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            # ROI for DeepFace (Logic Layer)
            face_roi = frame[y:y + h, x:x + w]

            # DeepFace Inference (Every 5 frames)
            if frame_count % 5 == 0:
                try:
                    raw_emotion, confidence = analyze_emotion(face_roi)

                    # Add raw prediction to the buffer
                    emotion_window.append(raw_emotion)

                    # CALCULATE STABLE EMOTION (The "Vote")
                    current_stable_emotion = get_dominant_emotion(emotion_window)

                    # No need to pass 'webcam' string anymore
                    log_emotion(
                        expression=current_stable_emotion,
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h)
                        # session_id = "webcam"
                        # session_ref_id will be auto-detected by db.py
                    )
                except Exception:
                    pass

    # 4.  --- DRAW THE SCI-FI HUD ---
            # Use specific color for emotion
            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)

            # Draw the custom tech UI (No MediaPipe needed!)
            draw_tech_ui(frame, x, y, w, h, hud_color, scan_pos)

            # Top Label (Solid Background for readability)
            label = f"{current_stable_emotion.upper()}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,0.8,
                2)

            # Label Background
            cv2.rectangle(frame, (x, y - 35),
                          (x + text_w + 10, y),
                          hud_color, -1)
            # Label Text
            cv2.putText(frame, label, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0), 2)

            """
            # DRAW HUD (Heads-Up Display)
            box_color = COLORS.get(current_stable_emotion,
                                   (0, 255, 0))

            # Bounding Box with "Corners" look
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Top Label (Emotion)
            cv2.rectangle(frame, (x, y -40), (x + w, y),
                          box_color, -1)  # Filled box
            cv2.putText(frame, f"{current_stable_emotion.upper()}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)  # Black text
            """
        # Save snapshot for dashboard
        save_snapshot(frame)

        cv2.imshow("FaceSense – Live (Mirror View)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")

if __name__ == "__main__":
    main()
