import cv2
import time

from expression_detector import FaceSense
from src.db import log_emotion
from src.models.emotion_model import EmotionModel
from utils.draw_results import draw_results
from utils.io_utils import save_snapshot

SESSION_ID = int(time.time())


def main():
    cap = cv2.VideoCapture(0)

    face_detector = FaceSense()
    emotion_model = EmotionModel("\PycharmProjects\FaceSense\src\models\emotion-ferplus.onnx")

    last_expression = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            face, bbox = face_detector.get_face(frame)

            if face is not None:
                label, confidence = emotion_model.predict(face)
                draw_results(frame, bbox, label, confidence)

                # snapshot
                save_snapshot(frame, tag="last")

                if label != last_expression:
                    save_snapshot(frame, tag=label)
                    last_expression = label

                log_emotion(label, confidence, bbox, session_id=SESSION_ID)

            cv2.imshow("FaceSense Live", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
