import cv2
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion

def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def main():
    cap = init_camera()
    frame_count = 0
    last_emotion = ""

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔁 MIRROR VIEW (horizontal flip)
        frame = cv2.flip(frame, 1)

        frame_count += 1
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Run emotion inference every 5 frames
            if frame_count % 5 == 0:
                try:
                    emotion, confidence = analyze_emotion(face_roi)
                    last_emotion = emotion

                    # No need to pass 'webcam' string anymore
                    log_emotion(
                        expression=emotion,
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h)
                        # session_id = "webcam"
                        # session_ref_id will be auto-detected by db.py
                    )
                except Exception:
                    pass

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw emotion label
            if last_emotion:
                cv2.putText(
                    frame,
                    last_emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        # Save snapshot for dashboard
        save_snapshot(frame)

        cv2.imshow("FaceSense – Live (Mirror View)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")

if __name__ == "__main__":
    main()
