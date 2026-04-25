import cv2
from pathlib import Path

CASCADE_PATH  = Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))




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