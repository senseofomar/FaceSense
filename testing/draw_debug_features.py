# utils/draw_debug_features.py
import cv2

def draw_debug_features(frame, mouth_w, lip_gap, curve, brow, start=(10,30), color=(255,255,255)):
    """
    Draw debug feature text on the frame.
    start: (x,y) top-left where text begins.
    """
    lines = [
        f"mouth_w: {mouth_w:.2f}",
        f"lip_gap: {lip_gap:.2f}",
        f"curve: {curve:.2f}",
        f"brow: {brow:.2f}"
    ]
    x, y = start
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
