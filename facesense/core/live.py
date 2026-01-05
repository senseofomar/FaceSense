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

        frame_count += 1
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            if frame_count % 5 == 0:
                try:
                    emotion, confidence = analyze_emotion(face_roi)
                    last_emotion = emotion
                    log_emotion(
                        expression=emotion,
                        confidence=confidence,
                        bbox=(x, y, x+w, y+h),
                        session_id="webcam"
                    )
                except Exception:
                    pass

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            if last_emotion:
                cv2.putText(
                    frame,
                    last_emotion,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,0,255),
                    2
                )

        save_snapshot(frame)
        cv2.imshow("FaceSense – Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
