import cv2
import time
from datetime import datetime
from collections import deque
from pathlib import Path

from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import (
    log_emotion, get_active_session,
    set_session_video_path
)

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
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

VOTE_BUFFER_SIZE = 10
AI_FRAME_SKIP    = 5

# ── Recordings folder (project_root/recordings/) ──────────────────────────────
RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)   # create folder if it doesn't exist


# ── VIDEO RECORDER ────────────────────────────────────────────────────────────
class VideoRecorder:
    """
    Wraps cv2.VideoWriter.
    - Starts automatically when a session becomes active.
    - Stops and finalises the file when the session ends.
    - One .avi file per session, named  session_<id>_<timestamp>.avi
    - Saves annotated frames (HUD + emotion label included).
    """

    def __init__(self):
        self._writer      = None   # cv2.VideoWriter instance
        self._session_id  = None   # which session we're recording for
        self._video_path  = None   # absolute path to the current file

    # ── public ───���────────────────────────────────────────────────────────────

    @property
    def is_recording(self):
        return self._writer is not None and self._writer.isOpened()

    @property
    def session_id(self):
        return self._session_id

    def update(self, active_session, frame_shape):
        """
        Call once per loop iteration with the current active session tuple
        (id, name) or None, and the shape of the video frame.
        Handles start / stop automatically.
        """
        current_id = active_session[0] if active_session else None

        if current_id and current_id != self._session_id:
            # New session started → open a new video file
            self._stop()
            self._start(current_id, active_session[1], frame_shape)

        elif not current_id and self._session_id:
            # Session ended → finalise file
            self._stop()

    def write(self, frame):
        """Write a single annotated frame. Safe to call even when not recording."""
        if self.is_recording:
            self._writer.write(frame)

    # ── private ───────────────────────────────────────────────────────────────

    def _start(self, session_id, session_name, frame_shape):
        h, w      = frame_shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"session_{session_id}_{timestamp}.avi"
        path      = str(RECORDINGS_DIR / filename)

        # MJPG codec — widely supported, good quality/size balance
        fourcc        = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer  = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
        self._session_id = session_id
        self._video_path = path

        # Store path in DB so dashboard can find the file later
        set_session_video_path(session_id, path)
        print(f"[Recorder] Started → {path}")

    def _stop(self):
        if self._writer is not None:
            self._writer.release()
            print(f"[Recorder] Saved  → {self._video_path}")
        self._writer     = None
        self._session_id = None
        self._video_path = None


# ── DRAW HELPERS ──────────────────────────────────────────────────────────────

def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap




def draw_hud(frame, x, y, w, h, color, scan_line_pos, confidence):
    """Draws the clean Sci-Fi brackets and scanning line."""
    ll = int(w * 0.15)  # Slightly shorter, sleeker corner brackets
    thick = 2

    # Corner Brackets
    cv2.line(frame, (x, y), (x + ll, y), color, thick)
    cv2.line(frame, (x, y), (x, y + ll), color, thick)
    cv2.line(frame, (x + w, y), (x + w - ll, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + ll), color, thick)
    cv2.line(frame, (x, y + h), (x + ll, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - ll), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w - ll, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - ll), color, thick)

    # Scanning Laser Line
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), color, 1)

    # Sleek Confidence Progress Bar (Positioned neatly above the box)
    bar_y = max(y - 8, 0)
    cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + 3), (40, 40, 40), -1)  # Dark background track
    cv2.rectangle(frame, (x, bar_y), (x + int(w * confidence), bar_y + 3), color, -1)  # Fill

def draw_label(frame, x, y, emotion, confidence, color):
    """Draws a modern, flat-design emotion label."""
    # Clean text formatting: e.g., "HAPPY • 70%"
    clean_emotion = emotion.upper()
    label = f"{clean_emotion} {int(confidence * 100)}%"

    font = cv2.FONT_HERSHEY_DUPLEX  # Slightly more modern font than standard SIMPLEX
    scale = 0.6
    thick = 1

    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    # Position box above the confidence bar
    y_box_bottom = max(y - 12, 0)
    y_box_top = max(y_box_bottom - th - 10, 0)

    # Draw flat background color box (no harsh black outline)
    cv2.rectangle(frame, (x, y_box_top), (x + tw + 10, y_box_bottom), color, -1)

    # Text Color Logic: If the background is bright (Yellow/Cyan/Green), use Black text. If dark, use White.
    # To keep it simple and readable for now, we'll stick to black text for high contrast.
    cv2.putText(frame, label, (x + 5, y_box_bottom - 5), font, scale, (0, 0, 0), thick)

def draw_system_stats(frame, fps, is_recording, frame_count):
    cv2.rectangle(frame, (10,10), (195,85), (0,0,0),   -1)
    cv2.rectangle(frame, (10,10), (195,85), COLOR_GREEN, 1)
    font, scale = cv2.FONT_HERSHEY_PLAIN, 1.05
    cv2.putText(frame, "SYSTEM: ONLINE",      (20,32), font, scale, COLOR_GREEN,   1)
    cv2.putText(frame, f"FPS:    {int(fps)}", (20,52), font, scale, (0,220,255),   1)
    if is_recording:
        if frame_count % 30 < 15:
            cv2.circle(frame, (25,68), 5, COLOR_RED, -1)
            cv2.putText(frame, "REC+LOG", (36,72), font, scale, COLOR_RED, 1)
    else:
        cv2.putText(frame, "LOGS: IDLE", (20,72), font, scale, COLOR_GRAY, 1)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    cap      = init_camera()
    recorder = VideoRecorder()

    frame_count            = 0
    emotion_window         = deque(maxlen=VOTE_BUFFER_SIZE)
    current_stable_emotion = "neutral"
    current_confidence     = 0.0
    scan_pos               = 0.0
    scan_direction         = 0.04
    prev_time              = 0.0
    active_session         = None   # cached (id, name)

    last_face      = None
    no_face_frames = 0
    NO_FACE_TOL    = 8

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

        # ── Check session every 30 frames ────────────────────────────────────
        if frame_count % 30 == 0:
            active_session = get_active_session()
            # Tell the recorder — it auto-starts/stops as needed
            recorder.update(active_session, frame.shape)

        is_recording_active = recorder.is_recording

        # ── Face detection ────────────────────────────────────────────────────
        raw_faces = detect_faces(frame)

        if len(raw_faces) > 0:
            best           = max(raw_faces, key=lambda r: r[2]*r[3])
            last_face      = best
            no_face_frames = 0
            active_face    = best
        elif no_face_frames < NO_FACE_TOL and last_face is not None:
            no_face_frames += 1
            active_face     = last_face
        else:
            active_face = None

        # ── AI + drawing ──────────────────────────────────────────────────────
        if active_face is None:
            cv2.putText(frame, "Searching...", (20, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)
        else:
            x, y, w, h = active_face
            face_roi    = frame[y:y+h, x:x+w]

            if frame_count % AI_FRAME_SKIP == 0:
                if face_roi.shape[0] >= 48 and face_roi.shape[1] >= 48:
                    try:
                        raw, conf = analyze_emotion(face_roi)
                        emotion_window.append(raw)
                        current_stable_emotion = get_dominant_emotion(emotion_window)
                        current_confidence     = conf
                        log_emotion(
                            expression=current_stable_emotion,
                            confidence=conf,
                            bbox=(x, y, x+w, y+h)
                        )
                    except Exception:
                        pass

            hud_color = COLORS.get(current_stable_emotion, COLOR_GREEN)
            draw_hud(frame, x, y, w, h, hud_color, scan_pos, current_confidence)
            draw_label(frame, x, y, current_stable_emotion, current_confidence, hud_color)

        draw_system_stats(frame, fps, is_recording_active, frame_count)

        # ── Write annotated frame to video if recording ───────────────────────
        recorder.write(frame)

        save_snapshot(frame)
        cv2.imshow("FaceSense - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up — make sure video is finalised even if user presses q mid-session
    recorder.update(None, (480, 640, 3))
    cap.release()
    cv2.destroyAllWindows()
    print("FaceSense stopped.")


if __name__ == "__main__":
    main()