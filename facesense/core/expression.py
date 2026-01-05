from deepface import DeepFace


def analyze_emotion(face_roi):
    result = DeepFace.analyze(
        face_roi,
        actions=["emotion"],
        enforce_detection=False
    )

    dominant = result[0]["dominant_emotion"]
    confidence = result[0]["emotion"][dominant] / 100.0
    return dominant, confidence
