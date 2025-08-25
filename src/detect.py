# src/detect.py
from __future__ import annotations
import cv2
from typing import List, Tuple

def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade at: {cascade_path}")
    return detector

def detect_faces(bgr_frame, detector: cv2.CascadeClassifier,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (40, 40)) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h) for each detected face.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return list(faces)

def draw_boxes(bgr_frame, faces, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in faces:
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, thickness)
    return bgr_frame
