import cv2
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion, get_active_session, save_video_path

# ── COLORS (BGR) ─────────────────────────────────────────────────────────────
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

VOTE_BUFFER_SIZE = 5
AI_FRAME_SKIP    = 5

# ── RECORDINGS FOLDER ────────────────────────────────────────────────────────
RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)   # create folder if it doesn't exist


# ── VIDEO RECORDER ───────────────────────────────────────────────────────────
class VideoRecorder:
    """
    Wraps cv2.VideoWriter.
    - Starts recording when a session becomes active
    - Writes every annotated frame (with HUD + emotion labels burned in)
    - Stops and finalises the file when the session ends
    - Saves the file path to the DB so the dashboard can find it
    """

    def __init__(self):
        self._writer      = None
        self._session_id  = None
        self._video_path  = None

    def start(self, session_id, session_name, frame_size):
        """Open a new video file for this session."""
        if self._writer is not None:
            self.stop()   # safety: close any previous writer

        # Build filename: recordings/session_3_john_2026-03-12_14-30-00.avi
        safe_name  = "".join(c if c.isalnum() else "_" for c in session_name)
        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename   = f"session_{session_id}_{safe_name}_{timestamp}.avi"
        filepath   = str(RECORDINGS_DIR / filename)

        # XVID codec — widely supported, good compression, works on Windows
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer     = cv2.VideoWriter(filepath, fourcc, 20.0, frame_size)
        self._session_id = session_id
        self._video_path = filepath

        # Persist path to DB immediately so dashboard can see it
        save_video_path(session_id, filepath)
        print(f"[REC] Recording started → {filepath}")

    def write(self, frame):
        """Write one frame. Safe to call even if not recording."""
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)

    def stop(self):
        """Finalise and close the video file."""
        if self._writer is not None:
            self._writer.release()
            print(f"[REC] Recording saved → {self._video_path}")
            self._writer     = None
            self._session_id = None
            self._video_path = None

    @property
    def is_recording(self):
        return self._writer is not None

    @property
    def active_session_id(self):
        return self._session_id


# ── HELPERS ───────────────────────────────────────────────────────────────────
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
    ll, thick = int(w * 0.18), 2
    cv2.line(frame, (x,     y),     (x+ll,   y),      color, thick)
    cv2.line(frame, (x,     y),     (x,       y+ll),   color, thick)
    cv2.line(frame, (x+w,   y),     (x+w-ll,  y),      color, thick)
    cv2.line(frame, (x+w,   y),     (x+w,     y+ll),   color, thick)
    cv2.line(frame, (x,     y+h),   (x+ll,    y+h),    color, thick)
    cv2.line(frame, (x,     y+h),   (x,        y+h-ll),color, thick)
    cv2.line(frame, (x+w,   y+h),   (x+w-ll,  y+h),   color, thick)
    cv2.line(frame, (x+w,   y+h),   (x+w,     y+h-ll),color, thick)
    scan_y = y + int(scan_line_pos * h)
    cv2.line(frame, (x, scan_y), (x+w, scan_y), color, 1)
    bar_y = max(y - 12, 0)
    cv2.rectangle(frame, (x, bar_y), (x+w,                bar_y+5), (40,40,40), -1)
    cv2.rectangle(frame, (x, bar_y), (x+int(w*confidence), bar_y+5), color,     -1)
    cv2.putText(frame, f"CNF:{int(confidence*100)}%",
                (x+w+5, y+20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)


def draw_label(frame, x, y, emotion, confidence, color):
    label = f"{EMOTION_PREFIX.get(emotion, emotion.upper())} {int(confidence*100)}%"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    y1 = max(y - th - 14, 0)
    cv2.rectangle(frame, (x,    y1), (x+tw+12, y-2), color,     -1)
    cv2.rectangle(frame, (x,    y1), (x+tw+12, y-2), (0,0,0),    1)
    cv2.putText(frame,   label, (x+6, y-7), font, scale, (0,0,0), thick)


def draw_system_stats(frame, fps, is_recording, frame_count):
    cv2.rectangle(frame, (10, 10), (195, 85), (0,0,0),   -1)
    cv2.rectangle(frame, (10, 10), (195, 85), COLOR_GREEN, 1)
    font, scale = cv2.FONT_HERSHEY_PLAIN, 1.05
    cv2.putText(frame, "SYSTEM: ONLINE",      (20, 32), font, scale, COLOR_GREEN,   1)
    cv2.putText(frame, f"FPS:    {int(fps)}", (20, 52), font, scale, (0,220,255),   1)
    if is_recording:
        if frame_count % 30 < 15:
            cv2.circle(frame, (25, 68), 5, COLOR_RED, -1)
            cv2.putText(frame, "LOGS: ACTIVE", (36, 72), font, scale, COLOR_RED,   1)
    else:
        cv2.putText(frame, "LOGS: IDLE", (20, 72), font, scale, COLOR_GRAY, 1)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    cap      = init_camera()
    recorder = VideoRecorder()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_w, frame_h)

    frame_count            = 0
    emotion_window         = deque(maxlen=VOTE_BUFFER_SIZE)
    current_stable_emotion = "neutral"
    current_confidence     = 0.0

    scan_pos            = 0.0
    scan_direction      = 0.04
    prev_time           = 0.0
    last_known_session  = None   # track session changes

    last_face      = None
    no_face_frames = 0
    NO_FACE_TOLERANCE = 8

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

        # ── SESSION / RECORDING STATE CHECK (every 30 frames) ────────────────
        if frame_count % 30 == 0:
            session = get_active_session()   # (id, name) or None

            if session and session != last_known_session:
                # New session just started — begin recording
                recorder.start(session[0], session[1], frame_size)
                last_known_session = session

            elif not session and last_known_session is not None:
                # Session just ended — finalise video
                recorder.stop()
                last_known_session = None

        # ── FACE DETECTION ────────────────────────────────────────────────────
        raw_faces = detect_faces(frame)
        if len(raw_faces) > 0:
            best           = max(raw_faces, key=lambda r: r[2] * r[3])
            last_face      = best
            no_face_frames = 0
            active_face    = best
        elif no_face_frames < NO_FACE_TOLERANCE and last_face is not None:
            no_face_frames += 1
            active_face = last_face
        else:
            no_face_frames += 1
            active_face = None

        # ── AI + DRAW ─────────────────────────────────────────────────────────
        if active_face is not None:
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
        else:
            cv2.putText(frame, "Searching...", (20, frame_h - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GRAY, 1)

        draw_system_stats(frame, fps, recorder.is_recording, frame_count)
        save_snapshot(frame)

        # ── WRITE TO VIDEO FILE ───────────────────────────────────────────────
        # Write the fully annotated frame (HUD + labels burned in)
        recorder.write(frame)

        cv2.imshow("FaceSense - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up — make sure video is finalised on quit
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("FaceSense stopped.")


if __name__ == "__main__":
    main()