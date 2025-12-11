# utils/io_utils.py
import os
import cv2
from datetime import datetime

os.makedirs("snapshots", exist_ok=True)

def save_snapshot(frame, tag="last", folder="snapshots"):
    """
    Save a snapshot image. Returns path.
    tag: 'last' overwrites last_frame.jpg, or use timestamp tag.
    """
    os.makedirs(folder, exist_ok=True)
    if tag == "last":
        out_path = os.path.join(folder, "last_frame.jpg")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(folder, f"{tag}_{ts}.jpg")
    # BGR -> JPG
    cv2.imwrite(out_path, frame)
    return out_path
