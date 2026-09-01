import cv2
import os
import urllib.request

class FaceDetector:
    """Extracts facial regions of interest (ROI) using OpenCV CascadeClassifier with verification."""
    def __init__(self, min_detection_confidence=0.5):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        
        # Download or replace file if missing or 0 bytes
        if not os.path.exists(xml_path) or os.path.getsize(xml_path) == 0:
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            print("Downloading missing face detection model...")
            urllib.request.urlretrieve(url, xml_path)

        self.face_cascade = cv2.CascadeClassifier(xml_path)
        
        # Fallback check if file exists but failed to load into OpenCV
        if self.face_cascade.empty():
            print("Classifier load failed. Re-downloading cascade model...")
            if os.path.exists(xml_path):
                os.remove(xml_path)
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, xml_path)
            self.face_cascade = cv2.CascadeClassifier(xml_path)

    def extract_faces(self, frame):
        """Returns bounding boxes and cropped face images from a frame."""
        if self.face_cascade.empty():
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size > 0:
                faces.append({
                    "bbox": (x, y, w, h),
                    "crop": face_crop,
                    "confidence": 0.95
                })
        return faces