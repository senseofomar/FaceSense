import cv2


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
