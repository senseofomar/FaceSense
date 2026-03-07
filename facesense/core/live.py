import cv2
import time
from collections import deque
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion, get_active_session

# ── CONFIG ──────────────────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)       # neutral   → #00FF00
COLOR_YELLOW = (0, 220, 255)     # happy     → #FFDC00
COLOR_RED    = (0, 0, 220)       # angry     → #DC0000
COLOR_BLUE   = (220, 100, 50)    # sad       → #3264DC
COLOR_CYAN   = (255, 200, 0)     # surprise  → #00C8FF
COLOR_PURPLE = (180, 0, 180)     # fear      → #B400B4
COLOR_DKGRN  = (0, 140, 0)      # disgust   → #008C00
COLOR_GRAY   = (100, 100, 100)

COLORS = {
    'neutral':  COLOR_GREEN,
    'happy':    COLOR_YELLOW,
    'angry':    COLOR_RED,
    'sad':      COLOR_BLUE,
    'surprise': COLOR_CYAN,
    'fear':     COLOR_PURPLE,
    'disgust':  COLOR_DKGRN,
}

# ASCII prefixes — no Unicode, OpenCV can't render emoji
EMOTION_PREFIX = {
    'happy':    '[HAPPY]',
    'sad':      '[SAD]',
    'angry':    '[ANGRY]',
    'surprise': '[SURPRISE]',
    'fear':     '[FEAR]',
    'disgust':  '[DISGUST]',
    'neutral':  '[NEUTRAL]',
}


def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap


def get_dominant_emotion(buffer):
    if not buffer:
        return "neutral"
    return max(set(buffer), key=buffer.count)


def draw_hud(frame, x, y, w, h, color, scan_line_pos, emotion, confidence):
    """Draws Sci-Fi HUD corner brackets, scan line, and confidence bar."""

    # 1. Corner Brackets
    ll = int(w * 0.18)
    thick = 2
    cv2.line(frame, (x,     y),     (x + ll, y),     color, thick)
    cv2.line(frame, (x,     y),     (x,      y + ll), color, thick)
    cv2.line(frame, (x + w, y),     (x + w - ll, y), color, thick)
    cv2.line(frame, (x + w, y),     (x + w, y + ll), color, thick)
    cv2.line(frame, (x,     y + h), (x + ll, y + h), color, thick)
    cv2.line(frame, (x,     y + h), (x,      y + h - ll), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w - ll, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w,  y + h - ll), color, thick)

    # 2. Scanning Laser Line
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), color, 1)

    # 3. Confidence Bar (sits just above the top bracket)
    bar_y = y - 12
    bar_h = 5
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_h), (40, 40, 40), -1)
    fill_w = int(w * confidence)
    cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + bar_h), color, -1)

    # 4. Side decorators
    cv2.putText(frame, f"CNF:{int(confidence * 100)}%",
                (x + w + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)


def draw_label(frame, x, y, emotion, confidence, color):
    """Draws the emotion label tag above the face box. No emoji — plain ASCII only."""
    prefix = EMOTION_PREFIX.get(emotion, '')
    label  = f"{prefix} {int(confidence * 100)}%"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    tag_x1 = x
    tag_y1 = y - th - 14
    tag_x2 = x + tw + 12
    tag_y2 = y - 2

    # Filled background, thin border, black text
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), (0, 0, 0), 1)
    cv2.putText(frame, label, (tag_x1 + 6, tag_y2 - 5), font, font_scale, (0, 0, 0), thickness)


def draw_system_stats(frame, fps, is_recording, frame_count):
    """Draws the top-left system HUD box."""
    cv2.rectangle(frame, (10, 10), (190, 85), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (190, 85), COLOR_GREEN, 1)

    font  = cv2.FONT_HERSHEY_PLAIN
    scale = 1.05

    cv2.putText(frame, "SYSTEM: ONLINE", (20, 32),  font, scale, COLOR_GREEN,  1)
    cv2.putText(frame, f"FPS:    {int(fps)}", (20, 52), font, scale, (0, 220, 255), 1)

    if is_recording:
        if frame_count % 30 < 15:
            cv2.circle(frame, (25, 68), 5, COLOR_RED, -1)
            cv2.putText(frame, "LOGS: ACTIVE", (36, 72), font, scale, COLOR_RED,  1)
    else:
        cv2.putText(frame, "LOGS: IDLE", (20, 72), font, scale, COLOR_GRAY, 1)


def main():
    cap = init_camera()
    frame_count   = 0
    emotion_window = deque(maxlen=7)
    current_stable_emotion = "neutral"
    current_confidence     = 0.0

    scan_pos       = 0.0
    scan_direction = 0.04
    prev_time      = 0.0
    is_recording_active = False

    print("FaceSense Live started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
        prev_time = curr_time

        scan_pos += scan_direction
        if scan_pos >= 1.0 or scan_pos <= 0.0:
            scan_direction *= -1

        faces = detect_faces(frame)

        # Ghost-buster: keep only the largest face
        if len(faces) > 0:
            largest = max(faces, key=lambda r: r[2] * r[3])
            faces = [largest]

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            if frame_count % 5 == 0:
                if frame_count % 30 == 0:
                    session = get_active_session()
                    is_recording_active = (session is not None)
                try:
                    raw, conf = analyze_emotion(face_roi)
                    emotion_window.append(raw)
                    current_stable_emotion = get_dominant_emotion(emotion_window)
                    current_confidence     = conf
                    log_emotion(
                        expression=current_stable_emotion,
                        confidence=conf,
                        bbox=(x, y, x + w, y + h)
                    )
                except Exception:
                    pass

            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)
            draw_hud(frame, x, y, w, h, hud_color, scan_pos,
                     current_stable_emotion, current_confidence)
            draw_label(frame, x, y, current_stable_emotion, current_confidence, hud_color)

        draw_system_stats(frame, fps, is_recording_active, frame_count)
        save_snapshot(frame)
        cv2.imshow("FaceSense - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("FaceSense stopped.")


if __name__ == "__main__":
    main()