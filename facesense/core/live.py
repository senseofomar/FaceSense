import cv2
from collections import deque # this is to add buffer to emotion, stable feed
import time
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


def draw_hud(frame, x, y, w, h, color, scan_line_pos, emotion, confidence):
    """Draws a Sci-Fi / Iron Man style HUD around the face + confidence bar"""

    # 1. Corner Brackets
    len_line = int(w * 0.15)
    thick = 2

    # Corners
    cv2.line(frame, (x, y), (x + len_line, y), color, thick)
    cv2.line(frame, (x, y), (x, y + len_line), color, thick)
    cv2.line(frame, (x + w, y), (x + w - len_line, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + len_line), color, thick)
    cv2.line(frame, (x, y + h), (x + len_line, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - len_line), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w - len_line, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - len_line), color, thick)

    # 2. Scanning Laser Line (Moves up and down)
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y),
             (0, 255, 0), 2)

    # 3. CONFIDENCE BAR (The "Health Bar")
    # Background (Gray)
    bar_x = x
    bar_y = y - 15
    bar_h = 5
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w, bar_y + bar_h), (50, 50, 50), -1)
    # Foreground (Color fill based on confidence)
    fill_width = int(w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_h), color, -1)

    # 4. Decorators
    cv2.circle(frame, (x + w // 2, y + h // 2), 2, COLOR_WHITE, -1)
    cv2.putText(frame, f"ID: 0x{id(x) % 1000:03X}", (x + w + 5, y + 20),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    cv2.putText(frame, f"CNF: {int(confidence * 100)}%", (x + w + 5, y + 40),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


def draw_system_stats(frame, fps):
    """Draws FPS and System Status in Top-Left"""
    # Background box
    cv2.rectangle(frame, (10, 10), (230, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (230, 90), (0, 255, 0), 1)

    # Text
    cv2.putText(frame, "SYSTEM: ONLINE",
                (25, 35), cv2.FONT_HERSHEY_PLAIN,
                1.2, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (25, 55), cv2.FONT_HERSHEY_PLAIN,
                1.2, (0, 255, 255), 1)
    cv2.putText(frame, "DB LOG: ACTIVE",
                (25, 75), cv2.FONT_HERSHEY_PLAIN,
                1.2, (0, 150, 255), 1)

def main():
    cap = init_camera()
    frame_count = 0

    # --- STABILIZATION BUFFER ---
    # Stores the last 7 emotions to prevent flickering
    emotion_window = deque(maxlen=7)
    current_stable_emotion = "neutral"
    current_confidence = 0.0

    # Scanner animation variables
    scan_pos = 0.0
    scan_direction = 0.05

    # FPS Calculation
    prev_time = 0

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Mirror
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

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
                    raw_emotion, conf = analyze_emotion(face_roi)

                    # Add raw prediction to the buffer
                    emotion_window.append(raw_emotion)

                    # CALCULATE STABLE EMOTION (The "Vote")
                    current_stable_emotion = get_dominant_emotion(emotion_window)

                    # Storing Confidence
                    current_confidence = conf
                    # No need to pass 'webcam' string anymore
                    log_emotion(
                        expression=current_stable_emotion,
                        confidence=conf,
                        bbox=(x, y, x + w, y + h)
                        # session_id = "webcam"
                        # session_ref_id will be auto-detected by db.py
                    )
                except Exception:
                    pass

            # --- DRAWING LOGIC ---
            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)

            # FLASH EFFECT: If very confident, flash white
            if current_confidence > 0.90:
                draw_color = COLOR_WHITE
            else:
                draw_color = hud_color

            # Draw the HUD Bar
            draw_hud(frame, x, y, w, h, draw_color, scan_pos,
                     current_stable_emotion, current_confidence)

            # Top Label (Solid Background for readability)
            label = f"{current_stable_emotion.upper()}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,0.8,
                2)

            # Label Background
            cv2.rectangle(frame, (x, y - 35),
                          (x + text_w + 10, y), hud_color, -1)
            # Label Text
            cv2.putText(frame, label, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0), 2)

        # Draw System Stats Overlay (New Feature)
        draw_system_stats(frame, fps)

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
