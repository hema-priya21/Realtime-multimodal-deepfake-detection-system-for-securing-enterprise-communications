import cv2

class FaceDetector:
    """Extracts facial regions of interest (ROI) using OpenCV built-in face detection."""
    def __init__(self, min_detection_confidence=0.5):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

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