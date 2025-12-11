# TensorFlow Debug LOG Spam Ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from expression_detector import FaceSense
from db import log_emotion
import time
from utils.io_utils import save_snapshot

SESSION_ID = int(time.time())

def draw_results(frame, bbox, emotion_label, confidence):
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 200, y1), (0, 255, 0), -1)

    # Draw text
    cv2.putText(frame,  f"{emotion_label} ({confidence*100:.0f}%)",
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2)

def main():
    detector = FaceSense()
    cap = cv2.VideoCapture(0)
    last_expression = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        # Get expression + bounding-box
        expression, confidence, bbox = detector.get_expression(frame)

        if bbox is None:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            draw_results(frame, bbox, expression, confidence)
            # Save last snapshot for dashboard
            save_snapshot(frame, tag ="last")
            # Optionally also save timestamped snapshot on expression change
            if expression != last_expression:
                save_snapshot(frame, tag=f"{expression}")
                last_expression = expression

            log_emotion(expression, confidence, bbox)

        cv2.imshow("FaceSense Live", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    main()