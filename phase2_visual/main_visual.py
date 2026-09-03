import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# ============================================================
# PROJECT PATHS & DEVICE CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUAL_MODEL_PATH = PROJECT_ROOT / "phase2_visual" / "visual_model.pth"
CASCADE_PATH = PROJECT_ROOT / "phase2_visual" / "haarcascades" / "haarcascade_frontalface_default.xml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visual model preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# LOAD TRAINED VISUAL MODEL
# ============================================================

def load_visual_model():
    model = models.mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)

    if VISUAL_MODEL_PATH.exists():
        model.load_state_dict(torch.load(str(VISUAL_MODEL_PATH), map_location=DEVICE))
        print(f"Loaded visual weights: {VISUAL_MODEL_PATH}")
    else:
        print(f"WARNING: Visual model not found at {VISUAL_MODEL_PATH}")

    model = model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# MULTIMODAL FUSION & ALERTING ENGINE
# ============================================================

def compute_fusion_score(visual_fake_prob: float, audio_fake_prob: float = 0.0) -> tuple[str, float]:
    """
    Combines visual and audio deepfake probabilities into a unified threat score.
    Weights: 70% Visual, 30% Audio (Adjustable based on Phase 3 integration).
    """
    visual_weight = 0.70
    audio_weight = 0.30

    # Calculated combined probability of frame being fake
    fused_score = (visual_fake_prob * visual_weight) + (audio_fake_prob * audio_weight)
    
    label = "FAKE" if fused_score >= 0.50 else "REAL"
    confidence = fused_score if label == "FAKE" else (1.0 - fused_score)
    
    return label, confidence * 100

def main():
    visual_model = load_visual_model()

    # Load local cascade XML
    if CASCADE_PATH.exists():
        xml_path = str(CASCADE_PATH)
    else:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(xml_path)

    if face_cascade.empty():
        print(f"ERROR: Could not load Haar Cascade XML from: {xml_path}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open video webcam stream.")
        return

    print("Multimodal Fusion active. Press 'q' inside the display window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture video frame.")
            break

        # Mirror video stream horizontally to fix camera alignment
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # Crop face region
            face_roi = frame[y : y + h, x : x + w]

            # Preprocess face crop for PyTorch visual model
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)
            input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

            # Visual model inference
            with torch.no_grad():
                outputs = visual_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                visual_fake_prob = probs[0].item()  # Class 0: FAKE, Class 1: REAL

            # Placeholder audio score (Replace with Phase 3 audio analysis output)
            audio_fake_prob = 0.0

            # Compute multimodal threat score
            label, score = compute_fusion_score(visual_fake_prob, audio_fake_prob)

            # UI Bounding Box and Overlay
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
            overlay_text = f"[FUSED] {label}: {score:.1f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, overlay_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Realtime Multimodal Deepfake Detection Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()