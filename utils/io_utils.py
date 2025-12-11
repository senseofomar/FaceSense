# utils/io_utils.py
import os
import streamlit as st

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

SNAPSHOT_PATH = os.path.join("snapshots", "last_frame.jpg")

def load_last_snapshot(path=SNAPSHOT_PATH):
    if not os.path.exists(path):
        # helpful debug info
        st.warning(f"No snapshot found at: {os.path.abspath(path)}. Run FaceSense to create snapshots.")
        return None
    img = cv2.imread(path)
    if img is None:
        st.error(f"Snapshot found but couldn't read image file: {os.path.abspath(path)}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# then in layout:
img = load_last_snapshot()
if img is not None:
    st.image(img, use_column_width=True)
