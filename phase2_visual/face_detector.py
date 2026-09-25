import os
import urllib.request

import cv2

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """Extracts facial regions of interest (ROI) using OpenCV's YuNet DNN
    face detector (cv2.FaceDetectorYN). Replaces the previous Haar cascade
    to improve detection on rotated/off-angle faces; the public interface
    (extract_faces -> list of {"bbox", "crop", "confidence"}) is unchanged
    so callers (web_app.py) require no changes.
    """
    def __init__(self, min_detection_confidence=0.5):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, MODEL_FILENAME)

        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            print("Downloading missing YuNet face detection model...")
            urllib.request.urlretrieve(MODEL_URL, model_path)

        # score_threshold / nms_threshold / top_k match the OpenCV Zoo
        # reference demo's own defaults for this model -- not independently
        # tuned, per instructions not to aggressively tune thresholds yet.
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.min_detection_confidence = min_detection_confidence

    def extract_faces(self, frame):
        """Returns bounding boxes and cropped face images from a frame."""
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))

        _, detections = self.detector.detect(frame)

        faces = []
        if detections is None:
            return faces

        for det in detections:
            score = float(det[14])
            if score < self.min_detection_confidence:
                continue

            x, y, w, h = det[0:4]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            faces.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "crop": face_crop,
                "confidence": score,
            })

        return faces


def select_largest_face(faces):
    """Picks the single face live inference would act on: the largest
    detected bounding box by area, avoiding a smaller face in the
    background instead of the person in front of the camera.

    Shared by web_app.py's live inference (run_visual_inference) and
    dataset_pipeline's YuNet-based training-data extraction
    (extract_frames_yunet.py), so both pipelines apply the exact same
    face-selection rule from one place instead of two copies that could
    silently drift apart. Returns None if `faces` is empty.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
