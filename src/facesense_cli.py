import sys
import os
import cv2
import numpy as np

# Import your chosen model wrapper here
from face_detector import FaceDetector
from models.emotion_model import EmotionModel # <-- Use this for your PyTorch VGG

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

    print("--- Loading Models ---")
    detector = FaceDetector()

    # Initialize model (adjust path if using PyTorch version)
    try:
        model = EmotionModel()  # Or EmotionModel("src/models/fer2013_vgg19.pth")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return

    print(f"--- Processing {os.path.basename(image_path)} ---")

    # 1. Detect (With Padding!)
    face, bbox = detector.detect(image, padding_pct=0.2)  # 20% padding

    if face is None:
        print("No face detected.")
        return

    # 2. Predict
    label, confidence, probs = model.predict(face)

    # 3. Output Text
    print(f"\nPrediction: {label.upper()}")
    print(f"Confidence: {confidence:.2f}%\n")

    if isinstance(probs, dict):  # Handle DeepFace dict output
        for em, score in probs.items():
            print(f"{em:<10} {score:.2f}")
    elif isinstance(probs, list):  # Handle PyTorch list output
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        for em, score in zip(emotions, probs):
            print(f"{em:<10} {score:.3f}")

    # 4. Output Visual (Optional: Save the debug image)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({confidence:.1f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_path = "debug_output.jpg"
    cv2.imwrite(output_path, image)
    print(f"\nDebug image saved to: {output_path}")


if __name__ == "__main__":
    main()