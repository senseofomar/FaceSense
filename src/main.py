import torch
import cv2
import os
import sys
from torchvision import transforms

# 1. PATH SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Import local modules
from models.load_model import FaceSenseVGG19
from face_detector import FaceDetector

# 2. DEVICE & MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = FaceDetector()
model = FaceSenseVGG19(num_classes=7)

# Adjust path to where your .pth is located inside src/models/
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fer2013_vgg19.pth')

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['net'])
model.to(device).eval()

# 3. FIXED LABEL MAPPING (Alphabetical Order to fix Happy/Sad flip)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 4. PREPROCESSING (Matching VGG19-BN Expectations)
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    # Normalizing to [-1, 1] range which is standard for BN layers
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_emotion(image_filename):
    # Images are in FaceSense/data/raw/
    root_dir = os.path.dirname(SCRIPT_DIR)
    image_path = os.path.join(root_dir, 'data', 'raw', image_filename)

    img = cv2.imread(image_path)
    if img is None:
        return f"❌ File not found: {image_path}"

    # Use the detector to find the face (Critical for accuracy)
    face_crop, _ = detector.detect(img, padding_pct=0.1)

    if face_crop is not None:
        # Convert to RGB and Transform
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img_tensor = data_transforms(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)

            emotion = labels[predicted.item()]
            return f"RESULT: {emotion} ({conf.item() * 100:.2f}%)"
    else:
        return "❌ Face detector could not find a face."


if __name__ == "__main__":
    # Ensure this file is in your data/raw folder
    test_img = "angry2.jpg"
    print(f"--- Running Inference on {test_img} ---")
    print(predict_emotion(test_img))