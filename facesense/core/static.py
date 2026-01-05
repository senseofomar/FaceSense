import cv2
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.storage.db import log_emotion

def run_on_image(image_path, log_to_db=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    faces = detect_faces(img)

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        emotion, confidence = analyze_emotion(face_roi)

        if log_to_db:
            log_emotion(
                expression=emotion,
                confidence=confidence,
                bbox=(x, y, x+w, y+h),
                session_id="static"
            )

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            img,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,0,255),
            2
        )

    cv2.imshow("FaceSense – Static", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_on_image("angryac.jpeg")
