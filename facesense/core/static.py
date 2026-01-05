import cv2
from pathlib import Path

from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.storage.db import log_emotion

def run_on_image(image_path, show = True, log_to_db=True):
    image_path = Path(image_path)

    if not image_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        image_path = project_root / image_path

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Invalid image path: {image_path}")

    faces = detect_faces(img)
    print(f"Faces detected: {len(faces)}")

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

    # --- SAVE PROCESSED IMAGE ---
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "storage" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(image_path).stem
    output_path = processed_dir / f"{input_name}_processed.jpg"

    cv2.imwrite(str(output_path), img)
    print(f"✅ Processed image saved to: {output_path}")

    # --- OPTIONAL DISPLAY ---
    if show:
        cv2.imshow("FaceSense – Static", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_on_image("facesense/storage/raw/angryac.jpeg")
