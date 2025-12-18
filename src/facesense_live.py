# now safe to import sibling packages
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# TensorFlow Debug LOG Spam Ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from expression_detector import FaceSense
from db import log_emotion
import time
from utils.io_utils import save_snapshot
from utils.append_features_csv import append_features_csv
from utils.draw_results import draw_results
from models.emotion_model import EmotionModel


import warnings
# Optional: suppress mediapipe/protobuf deprecation warning (harmless)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

SESSION_ID = int(time.time())


def main():
    # initialize webcam, detector, etc. (your existing setup)
    cap = cv2.VideoCapture(0)
    detector = FaceSense()
    emotion_model = EmotionModel("src/models/fer2013.onnx")

    # optionally start in debug OFF; press 'd' to toggle during runtime
    detector.set_debug(False)
    print("Debug mode is OFF. Press 'd' in the video window to toggle.")

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

                # --- LOG RAW FEATURES TO CSV (Task 2) ---
                try:
                    mw = detector.last_mouth_width
                    lg = detector.last_lip_gap
                    cv = detector.last_curve
                    br = detector.last_brow

                    row = [
                        time.time(),
                        expression,
                        confidence,
                        mw, lg, cv, br,
                        x1, y1, x2, y2
                    ]

                    append_features_csv(row)
                except Exception as e:
                    print("CSV logging error:", e)

                # log to DB (ensure log_emotion handles casting)
                try:
                    log_emotion(expression, confidence, bbox, session_id=SESSION_ID)
                except Exception as e:
                    print("DB logging error:", e)

            # show frame
            cv2.imshow("FaceSense Live", frame)
            key = cv2.waitKey(1) & 0xFF

            # press ESC to quit
            if key == 27:
                break

            # press 'd' to toggle debug mode
            if key == ord('d'):
                new = not detector.debug
                detector.set_debug(new)
                print(f"Debug mode set to: {detector.debug}")


    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ =="__main__":
    main()