# utils/snapshot.py
import os
import cv2
import time
from datetime import datetime


def save_snapshot(frame, tag="last", folder="snapshots"):
    folder = os.path.join(os.path.dirname(__file__), "../..", "snapshots")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    if tag == "last":
        final_path = os.path.join(folder, "last_frame.jpg")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(folder, f"{tag}_{ts}.jpg")

    # UNIQUE temp file (important on Windows)
    tmp_path = final_path.replace(
        ".jpg",
        f".{os.getpid()}.tmp.jpg"
    )

    # Write temp file
    ok = cv2.imwrite(tmp_path, frame)
    if not ok:
        raise RuntimeError(f"Failed to save snapshot to {tmp_path}")

    # Retry atomic replace (Windows-safe)
    for _ in range(5):
        try:
            os.replace(tmp_path, final_path)
            return final_path
        except PermissionError:
            time.sleep(0.05)

    # Last attempt (fail loudly)
    os.replace(tmp_path, final_path)
    return final_path


"""
Why this works

cv2.imwrite(tmp) → writes fully

os.replace(tmp, final) → instant swap

Streamlit never reads a partial file
"""