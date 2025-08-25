# src/main.py
from __future__ import annotations
import argparse
import os
import cv2

from detect import load_face_detector, detect_faces, draw_boxes
from classify import EmotionClassifier
from utils import crop_and_preprocess, put_label

def run_on_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read: {image_path}")

    detector = load_face_detector()
    clf = EmotionClassifier()

    faces = detect_faces(img, detector)
    chips = crop_and_preprocess(img, faces, target_size=(48, 48))
    labels = clf.predict(chips)

    # Draw boxes + labels
    for (rect, label) in zip(faces, labels):
        (x, y, w, h) = rect
        draw_boxes(img, [rect])
        put_label(img, label, x, y)

    cv2.imshow("FaceSense - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_webcam(cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {cam_index}")

    detector = load_face_detector()
    clf = EmotionClassifier()

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detect_faces(frame, detector)
        chips = crop_and_preprocess(frame, faces, target_size=(48, 48))
        labels = clf.predict(chips)

        for (rect, label) in zip(faces, labels):
            (x, y, w, h) = rect
            draw_boxes(frame, [rect])
            put_label(frame, label, x, y)

        cv2.imshow("FaceSense - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    ap = argparse.ArgumentParser(description="FaceSense - Expression Identification")
    ap.add_argument("--image", type=str, help="Path to an image to analyze")
    ap.add_argument("--webcam", type=int, nargs='?', const=0, help="Use webcam (default index 0)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.image:
        run_on_image(args.image)
    elif args.webcam is not None:
        run_on_webcam(args.webcam)
    else:
        print("Pass --image path/to.jpg or --webcam (optionally with index).")
