import sys
import os
import cv2

from face_detector import FaceDetector
from models.emotion_model import EmotionModel

MODEL_PATH = "src/models/fer2013_vgg19.pth"

def main():
    if len(sys.argv) != 2:
        print("Usage: python facesense_cli.py <image_path>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Error: Image not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read image.")
        return

    # --- Face detection ---
    detector = FaceDetector()
    face, bbox = detector.detect(image)

    if face is None:
        print("No face detected.")
        return

    # --- Emotion model ---
    model = EmotionModel(MODEL_PATH)
    label, confidence, probs = model.predict(face)

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Emotion: {label}")
    print(f"Confidence: {confidence:.2f}\n")

    print("Probabilities:")
    for k, v in probs.items():
        print(f"{k:<10} {v:.3f}")

if __name__ == "__main__":
    main()
