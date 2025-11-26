import cv2
from expression_detector import FaceSense

detector = FaceSense()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    expression, processed = detector.detect_expression(frame)

    cv2.putText(processed, f"Expression: {expression}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("FaceSense Live", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
