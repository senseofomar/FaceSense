import sys
import os
import cv2
import torch
from torchvision import transforms

# Import local modules
from face_detector import FaceDetector
from models.load_model import FaceSenseVGG19


def main():
    if len(sys.argv) != 2:
        print("Usage: python facesense_cli.py <image_path>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("--- Loading Models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Detector
    detector = FaceDetector()

    # 2. Initialize Model
    # Note: Ensure 'models/fer2013_vgg19.pth' exists relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'fer2013_vgg19.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model = FaceSenseVGG19(num_classes=7)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.to(device).eval()

    # Define Transforms (Same as static.py)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Labels
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    print(f"--- Processing {os.path.basename(image_path)} ---")

    # 3. Read Image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read image.")
        return

    # 4. Detect Face
    face_crop, bbox = detector.detect(image, padding_pct=0.15)

    if face_crop is None:
        print("No face detected.")
        return

    # 5. Predict
    img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    img_tensor = data_transforms(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)

        emotion_label = labels[predicted.item()]
        confidence_score = conf.item() * 100

    # 6. Output Results
    print(f"\nPrediction: {emotion_label.upper()}")
    print(f"Confidence: {confidence_score:.2f}%\n")

    # detailed probabilities
    print("Class Probabilities:")
    for idx, score in enumerate(probabilities[0]):
        print(f"{labels[idx]:<10} {score.item():.3f}")

    # 7. Visualization
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{emotion_label} ({confidence_score:.1f}%)"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_filename = "debug_output.jpg"
    cv2.imwrite(output_filename, image)
    print(f"\nSaved result to: {output_filename}")


if __name__ == "__main__":
    main()