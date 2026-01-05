import os
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


def load_image(image_path: str):
    """Load image from disk safely."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Failed to load image. Unsupported format.")

    print(f"✅ Image loaded successfully: {image_path}")
    return img

def detect_faces(img):
    """Detect faces using Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    print(f"👤 Faces detected: {len(faces)}")
    return faces


def analyze_emotion(img):
    """Run DeepFace emotion analysis."""
    print("🔍 Running emotion analysis...")

    results = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        enforce_detection=True
    )

    # DeepFace returns a list
    result = results[0]

    dominant_emotion = result["dominant_emotion"]
    emotion_scores = result["emotion"]

    print("\n🎭 Emotion Analysis Results")
    print("-" * 30)
    print(f"Dominant Emotion : {dominant_emotion}\n")

    for emotion, score in emotion_scores.items():
        print(f"{emotion.capitalize():10s}: {score:.2f}%")

    return dominant_emotion, emotion_scores




def draw_results(img, faces, emotion):
    """Draw bounding boxes and emotion label."""
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put emotion text above face
        cv2.putText(
            img,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return img


def display_image(img, title="Result"):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    IMAGE_PATH = "../../tu/angry2.jpg"

    image = load_image(IMAGE_PATH)
    faces = detect_faces(image)

    dominant_emotion, emotion_scores = analyze_emotion(image)

    output = draw_results(image, faces, dominant_emotion)
    display_image(output, title="FaceSense – Emotion Detection")
