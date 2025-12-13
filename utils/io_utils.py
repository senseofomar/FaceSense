# utils/io_utils.py
import os
import cv2
from datetime import datetime


def save_snapshot(frame, tag="last", folder="snapshots"):
    folder = os.path.join(os.path.dirname(__file__), "..", "snapshots")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    if tag == "last":
        final_path = os.path.join(folder, "last_frame.jpg")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(folder, f"{tag}_{ts}.jpg")
    tmp_path = final_path + ".tmp"
    #write to temp file first
    ok = cv2.imwrite(tmp_path, frame)
    if not ok:
        raise RuntimeError(f"Failed to save snapshot to {tmp_path}")

    #Atomic rename
    os.replace(tmp_path, final_path)

    return final_path


"""
Why this works

cv2.imwrite(tmp) → writes fully

os.replace(tmp, final) → instant swap

Streamlit never reads a partial file
"""