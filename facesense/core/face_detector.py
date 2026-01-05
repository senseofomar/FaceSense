import cv2
from pathlib import Path

CASCADE_PATH = Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml"

_face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
