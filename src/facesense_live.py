# TensorFlow Debug LOG Spam Ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from expression_detector import FaceSense
from db import log_emotion
import time
from utils.io_utils import save_snapshot
import warnings
# Optional: suppress mediapipe/protobuf deprecation warning (harmless)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

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

# at top of facesense_live.py (near other imports)


def main():
    # initialize webcam, detector, etc. (your existing setup)
    cap = cv2.VideoCapture(0)
    detector = FaceSense()

    # Initialize last expression variable INSIDE main (safe scope)
    last_expression = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            expression, confidence, bbox = detector.get_expression(frame)

            # draw + show - your existing drawing code here
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                draw_results(frame, bbox, expression, confidence)

                # Save the last snapshot (overwrites)
                try:
                    saved = save_snapshot(frame, tag="last")
                    print("Saved snapshot:", saved)
                except Exception as e:
                    print("Snapshot save error:", e)

                # Save a timestamped snapshot only when expression changes
                try:
                    if expression != last_expression:
                        saved_event = save_snapshot(frame, tag=expression)
                        print("Saved event snapshot:", saved_event)
                        last_expression = expression  # update the variable AFTER saving
                except Exception as e:
                    print("Event snapshot error:", e)

                # log to DB (ensure log_emotion handles casting)
                try:
                    log_emotion(expression, confidence, bbox, session_id=SESSION_ID)
                except Exception as e:
                    print("DB logging error:", e)

            # show frame
            cv2.imshow("FaceSense Live", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ =="__main__":
    main()