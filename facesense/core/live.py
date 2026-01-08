import cv2

from collections import deque # this is to add buffer to emotion, stable feed

import mediapipe as mp

from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion

# --- MEDIAPIPE CONFIGURATION ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mesh (Lightweight mode)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define colors (BGR format for OpenCV)
# We define this OUTSIDE the loop for efficiency
COLORS = {
    'angry': (0, 0, 255),    # Red
    'happy': (0, 255, 255),  # Yellow
    'sad': (255, 0, 0),      # Blue
    'neutral': (0, 255, 0),  # Green
    'surprise': (255, 255, 0), # Cyan
    'fear': (128, 0, 128),   # Purple
    'disgust': (0, 128, 0)   # Dark Green
}

def init_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def get_dominant_emotion(buffer):
    """Returns the most frequent emotion in the buffer (Stabilization)"""
    if not buffer:
        return "neutral"
    return max(set(buffer), key=buffer.count)

def main():
    cap = init_camera()
    frame_count = 0

    # --- STABILIZATION BUFFER ---
    # Stores the last 7 emotions to prevent flickering
    emotion_window = deque(maxlen=7)
    current_stable_emotion = "neutral"

    print("🎥 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Mirror and Convert for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. DRAW FACE MESH (The "Iron Man" Layer)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the intricate mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw contours (Eyes/Lips/Face Shape) for sharper look
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # 3. EMOTION LOGIC
        frame_count += 1
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            # ROI for DeepFace (Logic Layer)
            face_roi = frame[y:y + h, x:x + w]

            # Run emotion inference every 5 frames
            if frame_count % 5 == 0:
                try:
                    raw_emotion, confidence = analyze_emotion(face_roi)

                    # Add raw prediction to the buffer
                    emotion_window.append(raw_emotion)

                    # CALCULATE STABLE EMOTION (The "Vote")
                    current_stable_emotion = get_dominant_emotion(emotion_window)

                    # No need to pass 'webcam' string anymore
                    log_emotion(
                        expression=current_stable_emotion,
                        confidence=confidence,
                        bbox=(x, y, x + w, y + h)
                        # session_id = "webcam"
                        # session_ref_id will be auto-detected by db.py
                    )
                except Exception:
                    pass

            # 4. DRAW HUD (Heads-Up Display)
            box_color = COLORS.get(current_stable_emotion,
                                   (0, 255, 0))

            # Bounding Box with "Corners" look
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Top Label (Emotion)
            cv2.rectangle(frame, (x, y -40), (x + w, y),
                          box_color, -1)  # Filled box
            cv2.putText(frame, f"{current_stable_emotion.upper()}",
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)  # Black text

        # Save snapshot for dashboard
        save_snapshot(frame)

        cv2.imshow("FaceSense – Live (Mirror View)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped.")

if __name__ == "__main__":
    main()
