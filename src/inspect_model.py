import torch
import os
import sys

# Update this path to point to your actual .pth file
MODEL_PATH = "models/fer2013_vgg19.pth"


def check_model():
    if not os.path.exists(MODEL_PATH):
        print(f"File not found: {MODEL_PATH}")
        return

    print(f"--- Loading {MODEL_PATH} ---")
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Handle cases where the state dict is nested under "state_dict" or "net"
    if "net" in state_dict:
        print("Found 'net' key in dictionary. Using that.")
        state_dict = state_dict["net"]
    elif "state_dict" in state_dict:
        print("Found 'state_dict' key in dictionary. Using that.")
        state_dict = state_dict["state_dict"]

    print(f"\nTotal keys in weights file: {len(state_dict)}")

    # --- Check Classifier Structure ---
    print("\n--- Inspecting Classifier Layer ---")
    classifier_keys = [k for k in state_dict.keys() if "classifier" in k]

    if not classifier_keys:
        print("WARNING: No classifier keys found! The model might use 'fc' or 'head' instead.")
    else:
        for k in classifier_keys:
            shape = state_dict[k].shape
            print(f"Key: {k:<30} | Shape: {shape}")

    # --- Diagnose the Mismatch ---
    print("\n--- DIAGNOSIS ---")

    # SCENARIO 1: The "Simple" Classifier (What your current code expects)
    # Expectation: classifier.weight with shape [7, 512]
    has_simple = any(k == "classifier.weight" and state_dict[k].shape[1] == 512 for k in classifier_keys)

    # SCENARIO 2: The "Full" VGG Classifier (Standard VGG19)
    # Expectation: classifier.0.weight, classifier.3.weight, classifier.6.weight
    has_full = any("classifier.0.weight" in k for k in classifier_keys)

    if has_simple:
        print("✅ GOOD MATCH: The weights match your current 'Linear(512, 7)' code.")
    elif has_full:
        print("❌ MISMATCH: The file contains a FULL VGG classifier (4096 units).")
        print("   Your code is using a simplified 'Linear(512, 7)'.")
        print("   -> You need to update emotion_model.py to use the full VGG structure.")
    else:
        print("⚠️ UNKNOWN STRUCTURE: Compare the printed keys above with your code.")


if __name__ == "__main__":
    check_model()