# TensorFlow Debug LOG Spam Ignore
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import cv2
from expression_detector import FaceSense

detector = FaceSense()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    exp = detector.get_expression(frame)

    if exp:
        cv2.putText(frame, exp, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    cv2.imshow("FaceSense Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
