import cv2
import time
from collections import deque
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion, get_active_session

# ── COLOR CONFIG (BGR) ───────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED    = (0, 0, 220)
COLOR_BLUE   = (220, 100, 50)
COLOR_CYAN   = (255, 200, 0)
COLOR_PURPLE = (180, 0, 180)
COLOR_DKGRN  = (0, 140, 0)
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

EMOTION_PREFIX = {
    'happy':    '[HAPPY]',
    'sad':      '[SAD]',
    'angry':    '[ANGRY]',
    'surprise': '[SURPRISE]',
    'fear':     '[FEAR]',
    'disgust':  '[DISGUST]',
    'neutral':  '[NEUTRAL]',
}

# ── TUNING KNOBS ─────────────────────────────────────────────────────────────
# How many frames to smooth over. Lower = reacts faster, Higher = more stable
#   3 → snappy, can flicker a little
#   5 → balanced (recommended)
#   7 → very stable, hard to trigger subtle emotions like sad/fear
VOTE_BUFFER_SIZE = 5

# Run DeepFace every N frames (DeepFace is slow — don't call every frame)
#   3 → more responsive, needs good GPU/CPU
#   5 → balanced (recommended)
#   8 → slower response but lightest CPU load
AI_FRAME_SKIP = 5


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


def draw_hud(frame, x, y, w, h, color, scan_line_pos, confidence):
    """Corner brackets, scan line, confidence bar."""
    ll    = int(w * 0.18)
    thick = 2

    cv2.line(frame, (x,     y),     (x + ll,  y),       color, thick)
    cv2.line(frame, (x,     y),     (x,        y + ll),  color, thick)
    cv2.line(frame, (x + w, y),     (x + w - ll, y),    color, thick)
    cv2.line(frame, (x + w, y),     (x + w,   y + ll),  color, thick)
    cv2.line(frame, (x,     y + h), (x + ll,  y + h),   color, thick)
    cv2.line(frame, (x,     y + h), (x,        y+h-ll), color, thick)
    cv2.line(frame, (x + w, y + h), (x+w-ll,  y + h),  color, thick)
    cv2.line(frame, (x + w, y + h), (x + w,   y+h-ll), color, thick)

    # Scan line
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), color, 1)

    # Confidence bar above box
    bar_y = y - 12
    cv2.rectangle(frame, (x,        bar_y), (x + w,               bar_y + 5), (40, 40, 40), -1)
    cv2.rectangle(frame, (x,        bar_y), (x + int(w*confidence), bar_y + 5), color,      -1)

    # Side text
    cv2.putText(frame, f"CNF:{int(confidence*100)}%",
                (x + w + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)


def draw_label(frame, x, y, emotion, confidence, color):
    """Clean ASCII label tag above the face — no emoji."""
    prefix = EMOTION_PREFIX.get(emotion, f'[{emotion.upper()}]')
    label  = f"{prefix} {int(confidence * 100)}%"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness  = 2

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tag_y1 = max(y - th - 14, 0)   # clamp so it doesn't go off screen top

    cv2.rectangle(frame, (x, tag_y1),    (x + tw + 12, y - 2), color,       -1)
    cv2.rectangle(frame, (x, tag_y1),    (x + tw + 12, y - 2), (0, 0, 0),    1)
    cv2.putText(frame,   label, (x + 6, y - 7), font, font_scale, (0, 0, 0), thickness)


def draw_system_stats(frame, fps, is_recording, frame_count):
    cv2.rectangle(frame, (10, 10), (195, 85), (0, 0, 0),    -1)
    cv2.rectangle(frame, (10, 10), (195, 85), COLOR_GREEN,   1)

    font  = cv2.FONT_HERSHEY_PLAIN
    scale = 1.05

    cv2.putText(frame, "SYSTEM: ONLINE",    (20, 32), font, scale, COLOR_GREEN,        1)
    cv2.putText(frame, f"FPS:    {int(fps)}", (20, 52), font, scale, (0, 220, 255),    1)

    if is_recording:
        if frame_count % 30 < 15:
            cv2.circle(frame, (25, 68), 5, COLOR_RED, -1)
            cv2.putText(frame, "LOGS: ACTIVE", (36, 72), font, scale, COLOR_RED,  1)
    else:
        cv2.putText(frame, "LOGS: IDLE", (20, 72), font, scale, COLOR_GRAY, 1)


def main():
    cap = init_camera()

    frame_count  = 0
    emotion_window         = deque(maxlen=VOTE_BUFFER_SIZE)
    current_stable_emotion = "neutral"
    current_confidence     = 0.0

    scan_pos       = 0.0
    scan_direction = 0.04
    prev_time      = 0.0
    is_recording_active    = False

    # Track last known face position — used to suppress flicker
    last_face = None          # (x, y, w, h)
    no_face_frames = 0        # consecutive frames with no detection
    NO_FACE_TOLERANCE = 8     # hold last face for this many frames before clearing
                              # Raise this if face disappears too easily when you tilt head

    print("FaceSense Live started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
        prev_time = curr_time

        scan_pos += scan_direction
        if scan_pos >= 1.0 or scan_pos <= 0.0:
            scan_direction *= -1

        faces = detect_faces(frame)

        # Ghost-buster: keep only the single largest detected face
        if len(faces) > 0:
            largest   = max(faces, key=lambda r: r[2] * r[3])
            faces     = [largest]
            last_face = largest
            no_face_frames = 0
        else:
            no_face_frames += 1
            # Hold the last known face for a few frames before giving up
            # This eliminates 1-2 frame flicker from head movement
            if no_face_frames <= NO_FACE_TOLERANCE and last_face is not None:
                faces = [last_face]
            else:
                last_face = None

        if len(faces) == 0:
            # No face at all — draw a subtle message, keep last emotion on screen
            cv2.putText(frame, "Searching...", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)
        else:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y + h, x:x + w]

            # ── AI LOGIC ─────────────────────────────────────────────────────
            # Only run DeepFace every AI_FRAME_SKIP frames — it's expensive
            if frame_count % AI_FRAME_SKIP == 0:

                # Check recording status every 30 frames (DB call is slow)
                if frame_count % 30 == 0:
                    session = get_active_session()
                    is_recording_active = (session is not None)

                # Guard: face crop must be big enough for DeepFace to work well
                # If crop is tiny, DeepFace returns garbage (usually neutral/disgust)
                if face_roi.shape[0] >= 48 and face_roi.shape[1] >= 48:
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
                        pass  # keep last known emotion on failure

            # ── DRAWING LOGIC ─────────────────────────────────────────────────
            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)
            draw_hud(frame, x, y, w, h, hud_color, scan_pos, current_confidence)
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