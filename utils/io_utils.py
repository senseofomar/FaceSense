# utils/io_utils.py
import os
import cv2
from datetime import datetime

def save_snapshot(frame, tag="last", folder="snapshots"):
    folder = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    if tag == "last":
        out_path = os.path.join(folder, "last_frame.jpg")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(folder, f"{tag}_{ts}.jpg")
    ok = cv2.imwrite(out_path, frame)
    if not ok:
        raise RuntimeError(f"Failed to save snapshot to {out_path}")
    return os.path.abspath(out_path)
