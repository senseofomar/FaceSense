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


def display_image(img):
    """Display image using matplotlib."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()


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


if __name__ == "__main__":
    IMAGE_PATH = "disgust1.png"   # or full path if needed

    image = load_image(IMAGE_PATH)
    display_image(image)
    analyze_emotion(image)
