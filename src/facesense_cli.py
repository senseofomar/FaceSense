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
    model = EmotionModel("models/emotion-ferplus.onnx")

    face, bbox = detector.detect(image)

    if face is None:
        print("No face detected.")
        sys.exit(0)

    label, confidence, probs = model.predict(face)

    print(f"\nImage: {os.path.basename(image_path)}")
    print("Face detected: yes")
    print(f"Emotion: {label}")
    print(f"Confidence: {confidence:.2f}\n")

    print("Probabilities:")
    for name, p in zip(model.EMOTIONS if hasattr(model, "EMOTIONS") else [
        "Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear"
    ], probs):
        print(f"{name:<10} {p:.2f}")


if __name__ == "__main__":
    main()
