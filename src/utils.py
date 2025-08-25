# src/utils.py
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def crop_and_preprocess(bgr_frame,
                        faces: List[Tuple[int, int, int, int]],
                        target_size: Tuple[int, int] = (48, 48),
                        normalize: bool = True) -> np.ndarray:
    """
    For each face bounding box, crop → grayscale → resize → normalize.
    Returns (N, H, W) float32 array in [0,1] if normalize=True.
    """
    chips = []
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        # Add a small margin around the face box
        m = int(0.15 * max(w, h))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(gray.shape[1], x + w + m), min(gray.shape[0], y + h + m)
        face = gray[y0:y1, x0:x1]

        if face.size == 0:
            continue

        face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        face = face.astype("float32")
        if normalize:
            face /= 255.0
        chips.append(face)

    if not chips:
        return np.empty((0, target_size[0], target_size[1]), dtype="float32")

    return np.stack(chips, axis=0)

def put_label(bgr_frame, text: str, x: int, y: int):
    # Draw filled background for better readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(bgr_frame, (x, y - th - 6), (x + tw + 4, y), (0, 0, 0), -1)
    cv2.putText(bgr_frame, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
