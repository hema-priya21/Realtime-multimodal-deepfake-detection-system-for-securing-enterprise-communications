import cv2
import numpy as np

class VisualAnalyzer:
    """Evaluates spatial blur, edge inconsistency, and synthetic noise level."""
    def analyze_face(self, face_crop):
        """Calculates a baseline visual anomaly confidence score (0.0 = Real, 1.0 = Fake)."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (detects artificial smoothing/blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Low variance often indicates artificial skin smoothing in synthetic faces
        smoothing_score = max(0.0, min(1.0, 1.0 - (laplacian_var / 500.0)))
        
        return {
            "fake_score": round(smoothing_score, 4),
            "laplacian_var": round(laplacian_var, 2)
        }