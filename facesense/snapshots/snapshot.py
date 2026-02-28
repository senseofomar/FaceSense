# utils/snapshot.py
import os
import cv2
import time
from datetime import datetime



"""
Why this works

cv2.imwrite(tmp) → writes fully

os.replace(tmp, final) → instant swap

Streamlit never reads a partial file
"""