import sys
import os
import cv2

from face_detector import FaceDetector
from models.emotion_model import EmotionModel


def main():
    if len(sys.argv) != 2:
        print("Usage: python facesense_cli.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("Error: Image not found.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read image.")
        sys.exit(1)

    detector = FaceDetector()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion-ferplus.onnx")

    model = EmotionModel(MODEL_PATH)

    face, bbox = detector.detect(image)

    if face is None:
        print("No face detected.")
        sys.exit(0)

    label, confidence, probs = model.predict(face)

    if label is None:
        print("Emotion prediction failed.")
        return

    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{label} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    os.makedirs("data/processed", exist_ok=True)
    out_path = os.path.join(
        "data/processed",
        os.path.basename(image_path)
    )

    cv2.imwrite(out_path, image)

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Emotion: {label}")
    print(f"Confidence: {confidence:.2f}\n")

    print("Probabilities:")
    for k, v in probs.items():
        print(f"{k:<10} {v:.3f}")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
