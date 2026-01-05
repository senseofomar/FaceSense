import cv2
from deepface import DeepFace
from facesense.snapshots.snapshot import save_snapshot
from facesense.data.db import log_emotion



def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("❌ Cannot open webcam")
    return cap


def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def main():
    cap = init_camera()
    face_cascade = load_face_detector()

    frame_count = 0

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count +=1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Run DeepFace every 5 frames only
            if frame_count % 5 == 0:
                try:
                    result = DeepFace.analyze(
                        face_roi,
                        actions=["emotion"],
                        enforce_detection=False
                    )

                    dominant_emotion = result[0]["dominant_emotion"]
                    confidence = result[0]["emotion"][dominant_emotion] / 100.0

                    bbox = (x, y, x + w, y + h)

                    log_emotion(
                        expression=dominant_emotion,
                        confidence=confidence,
                        bbox=bbox,
                        session_id="webcam"
                    )

                except Exception:
                    continue

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw emotion label
            cv2.putText(
                frame,
                dominant_emotion if "dominant_emotion" in locals() else "",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        # Save latest frame (for dashboard)
        save_snapshot(frame, tag="last")

        cv2.imshow("FaceSense – Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")


if __name__ == "__main__":
    main()
