import cv2
from pathlib import Path

CASCADE_PATH  = Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,   # was 4 → raised to 5: kills flickering & ghost faces
                          # If your face stops being detected, lower to 6 or 5
        minSize=(77, 77)  # was 70 → raised: ignores far-away/partial detections
    )

"""
TUNING GUIDE
────────────────────────────────────────────────────────
minNeighbors   | Effect
──────────────────────────────
4  (too low)   | Flickers, ghost faces on background
6  (balanced)  | Good for well-lit static scenes
7  (current)   | Stable, needs clear frontal face
9  (too strict)| Misses real faces if you tilt head

minSize        | Effect
──────────────────────────────
(60,60)        | Detects far/small faces, more ghosts
(90,90)        | Current — ignores background noise
(120,120)      | Only detects close-up faces
────────────────────────────────────────────────────────
"""