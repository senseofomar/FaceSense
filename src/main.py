import torch
import cv2
import os
import sys

# 1. SETUP PATHS (Portably)
# This is the 'src' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This is the 'FaceSense' root folder
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Path to weights: FaceSense/src/models/fer2013_vgg19.pth
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fer2013_vgg19.pth')
# Path to images: FaceSense/data/raw/
IMAGE_DIR = os.path.join(ROOT_DIR, 'data', 'raw')

# Import your model class
from models.load_model import FaceSenseVGG19

# 2. SETUP DEVICE AND MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceSenseVGG19(num_classes=7)

# 3. LOAD WEIGHTS
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Weights not found at {MODEL_PATH}")
    print("Check if the .pth file is actually inside src/models/")
    sys.exit()

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['net'])
model.to(device)
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_emotion(image_filename):
    full_path = os.path.join(IMAGE_DIR, image_filename)
    img = cv2.imread(full_path)

    if img is None:
        print(f"❌ Image not found: {full_path}")
        return

    # Standard Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    img_tensor = torch.from_numpy(gray).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return labels[predicted.item()]


if __name__ == "__main__":
    # Ensure there is a file named 'test.jpg' in FaceSense/data/raw/
    test_img = "fear1.jpg"
    result = predict_emotion(test_img)
    if result:
        print(f"Prediction for {test_img}: {result}")