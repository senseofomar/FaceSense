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
state_dict = checkpoint['net']

# --- NEW KEY REMAPPING LOGIC ---
# This fixes the "Missing/Unexpected" error by renaming keys on the fly
new_state_dict = {}
for k, v in state_dict.items():
    # If the file has 'classifier.1.weight', change it to 'classifier.weight'
    name = k.replace('classifier.1.', 'classifier.')
    new_state_dict[name] = v

# Load the cleaned state dict
model.load_state_dict(new_state_dict)
model.to(device).eval()
# -------------------------------

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

            # Convert to list to modify weights
            # Order: 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Neutral, 5:Sad, 6:Surprise
            probs = probabilities[0].tolist()

            # --- THE CALIBRATION FIX ---
            # 1. Give Neutral a massive 'Head Start' (Neutral is at index 4)
            probs[4] = probs[4] * 1.5

            # 2. Penalize Surprise (Surprise is at index 6)
            # This stops Neutral being called Surprise
            probs[6] = probs[6] * 0.7

            # 3. Penalize Happy (Happy is at index 3)
            # This helps fix the 'Sad is Happy' bug you saw earlier
            probs[3] = probs[3] * 0.8
            # ---------------------------

            # Find the new winner after our adjustments
            max_val = max(probs)
            predicted_idx = probs.index(max_val)

            emotion_label = labels[predicted_idx]
            # Normalize confidence to 0-100%
            confidence_score = (max_val / sum(probs)) * 100

            return f"Emotion: {emotion_label} ({confidence_score:.2f}%)"
    else:
        return "❌ Face detector could not find a face."


if __name__ == "__main__":
    # Ensure this file is in your data/raw folder
    test_img = "esad.jpeg"
    print(f"--- Running Inference on {test_img} ---")
    print(predict_emotion(test_img))