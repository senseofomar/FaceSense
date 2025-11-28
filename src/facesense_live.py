# TensorFlow Debug LOG Spam Ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from expression_detector import FaceSense


def draw_results(frame, bbox, emotion_label):
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label background
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + 150, y1), (0, 255, 0), -1)

    # Draw text
    cv2.putText(frame, emotion_label, (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


detector = FaceSense()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    expression, bbox = detector.get_expression(frame)

    if bbox is not None:  # means face found
        draw_results(frame, bbox, expression)

    cv2.imshow("FaceSense Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
