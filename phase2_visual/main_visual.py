import cv2
import os
import urllib.request

class FaceDetector:
    """Extracts facial regions of interest (ROI) using OpenCV CascadeClassifier with auto-download fallback."""
    def __init__(self, min_detection_confidence=0.5):
        # Define local path for cascade XML inside phase2_visual directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
        
        # Auto-download cascade XML if not found locally
        if not os.path.exists(xml_path):
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            print("Downloading face detection model file...")
            urllib.request.urlretrieve(url, xml_path)

        self.face_cascade = cv2.CascadeClassifier(xml_path)

    def extract_faces(self, frame):
        """Returns bounding boxes and cropped face images from a frame."""
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