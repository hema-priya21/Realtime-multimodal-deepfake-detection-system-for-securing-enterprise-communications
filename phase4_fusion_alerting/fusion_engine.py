class MultimodalFusionEngine:
    """Fuses visual and audio anomaly confidence scores into an enterprise alert system."""
    def __init__(self, visual_weight=0.6, audio_weight=0.4, threshold=0.65):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.threshold = threshold

    def evaluate_threat(self, visual_score, audio_score):
        """Computes weighted deepfake threat score and triggers security status."""
        composite_score = (visual_score * self.visual_weight) + (audio_score * self.audio_weight)
        composite_score = round(composite_score, 4)

        if composite_score >= self.threshold:
            status = "CRITICAL ALERT: DEEPFAKE DETECTED"
            alert_level = "HIGH"
        elif composite_score >= 0.4:
            status = "WARNING: ANOMALOUS STREAM"
            alert_level = "MEDIUM"
        else:
            status = "AUTHENTIC"
            alert_level = "LOW"

        return {
            "composite_score": composite_score,
            "status": status,
            "alert_level": alert_level
        }