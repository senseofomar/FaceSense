import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def detect(self, image_bgr):
        """
        Returns:
        - face_crop (BGR)
        - bbox (x1, y1, x2, y2)
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None, None

        h, w, _ = image_bgr.shape
        det = results.detections[0]
        box = det.location_data.relative_bounding_box

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = image_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None, None

        return face, (x1, y1, x2, y2)
