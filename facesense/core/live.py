import cv2
import time
from collections import deque
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion, get_active_session

# --- CONFIG ---
COLOR_CYAN = (255, 255, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (100, 100, 100)

COLORS = {
    'angry': COLOR_RED,
    'happy': (0, 255, 255),  # Yellow
    'sad': (255, 0, 0),  # Blue
    'neutral': COLOR_GREEN,
    'surprise': COLOR_CYAN,
    'fear': (128, 0, 128),  # Purple
    'disgust': (0, 128, 0)  # Dark Green
}


def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap





def draw_hud(frame, x, y, w, h, color, scan_line_pos, emotion, confidence):
    """Draws Sci-Fi HUD + Confidence Bar (CLEAN VERSION)"""

    # 1. Corner Brackets
    len_line = int(w * 0.15)
    thick = 2

    cv2.line(frame, (x, y), (x + len_line, y), color, thick)
    cv2.line(frame, (x, y), (x, y + len_line), color, thick)
    cv2.line(frame, (x + w, y), (x + w - len_line, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + len_line), color, thick)
    cv2.line(frame, (x, y + h), (x + len_line, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - len_line), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w - len_line, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - len_line), color, thick)

    # 2. Scanning Laser Line
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), (0, 255, 0), 2)

    # 3. CONFIDENCE BAR
    bar_x = x
    bar_y = y - 15
    bar_h = 5
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w, bar_y + bar_h), (50, 50, 50), -1)
    fill_width = int(w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h), color, -1)

    # 4. Decorators (Removed Center Dot)
    cv2.putText(frame, f"ID: 0x{id(x) % 1000:03X}", (x + w + 5, y + 20),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    cv2.putText(frame, f"CNF: {int(confidence * 100)}%", (x + w + 5, y + 40),
                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


def draw_system_stats(frame, fps, is_recording, frame_count):
    """Draws FPS and Recording Status (Fixed Spacing)"""
    # Background box - Tightened height (100 -> 90) to look cleaner
    cv2.rectangle(frame, (10, 10), (220, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (220, 90), (0, 255, 0), 1)

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1.1

    # Fixed Spacing: 35, 55, 75 (Exactly 20px gap each)
    cv2.putText(frame, "SYSTEM: ONLINE", (20, 35), font, scale, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 55), font, scale, (0, 255, 255), 1)

    # DYNAMIC LOGGING STATUS
    if is_recording:
        if frame_count % 30 < 15:  # Blink
            cv2.circle(frame, (28, 70), 5, COLOR_RED, -1)
            cv2.putText(frame, "LOGS: ACTIVE", (40, 75), font, scale, COLOR_RED, 1)
    else:
        cv2.putText(frame, "LOGS: IDLE", (20, 75), font, scale, COLOR_GRAY, 1)


def main():
    cap = init_camera()
    frame_count = 0
    emotion_window = deque(maxlen=7)
    current_stable_emotion = "neutral"
    current_confidence = 0.0

    # State Variables
    scan_pos = 0.0
    scan_direction = 0.05
    prev_time = 0
    is_recording_active = False

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Update Scanner
        scan_pos += scan_direction
        if scan_pos >= 1.0 or scan_pos <= 0.0:
            scan_direction *= -1

        faces = detect_faces(frame)

        # --- GHOST BUSTER LOGIC ---
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            faces = [largest_face]

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            # --- AI LOGIC ---
            if frame_count % 5 == 0:
                if frame_count % 30 == 0:
                    session = get_active_session()
                    is_recording_active = (session is not None)

                try:
                    raw, conf = analyze_emotion(face_roi)
                    emotion_window.append(raw)
                    current_stable_emotion = get_dominant_emotion(emotion_window)
                    current_confidence = conf

                    log_emotion(
                        expression=current_stable_emotion,
                        confidence=conf,
                        bbox=(x, y, x + w, y + h)
                    )
                except:
                    pass

            # --- DRAWING LOGIC ---
            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)

            # Removed White Flash Logic - Just use the Emotion Color
            draw_hud(frame, x, y, w, h, hud_color, scan_pos, current_stable_emotion, current_confidence)

            # Label
            label = f"{current_stable_emotion.upper()}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            cv2.rectangle(frame, (x, y - 35), (x + text_w + 10, y), hud_color, -1)
            cv2.putText(frame, label, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        draw_system_stats(frame, fps, is_recording_active, frame_count)

        save_snapshot(frame)
        cv2.imshow("FaceSense – Live (Mirror View)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")


if __name__ == "__main__":
    main()