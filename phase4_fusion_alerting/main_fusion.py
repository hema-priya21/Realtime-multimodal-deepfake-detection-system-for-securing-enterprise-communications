import cv2
import sounddevice as sd
import numpy as np
import threading
import sys
import os

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase1_ingestion.buffer_manager import FrameBufferManager
from phase2_visual.face_detector import FaceDetector
from phase2_visual.visual_analyzer import VisualAnalyzer
from phase3_audio.audio_analyzer import AudioAnalyzer
from phase4_fusion_alerting.fusion_engine import MultimodalFusionEngine

# Shared state variables for audio thread synchronization
latest_audio_score = 0.0
running = True

def audio_stream_worker():
    """Background thread to sample microphone without blocking UI rendering."""
    global latest_audio_score, running
    analyzer = AudioAnalyzer(sample_rate=16000)
    chunk_size = int(16000 * 0.5)  # 0.5s audio chunks
    
    while running:
        try:
            audio_chunk = sd.rec(chunk_size, samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            metrics = analyzer.analyze_chunk(audio_chunk.flatten())
            latest_audio_score = metrics["fake_score"]
        except Exception:
            break

if __name__ == "__main__":
    stream = FrameBufferManager(src=0).start()
    detector = FaceDetector()
    visual_analyzer = VisualAnalyzer()
    fusion_engine = MultimodalFusionEngine(visual_weight=0.6, audio_weight=0.4, threshold=0.65)

    # Launch background thread for live microphone listening
    audio_thread = threading.Thread(target=audio_stream_worker, daemon=True)
    audio_thread.start()

    print("Multimodal Deepfake Detection Engine Active... Press 'q' to quit.")

    threat = {"status": "AUTHENTIC", "composite_score": 0.0, "alert_level": "LOW"}

    while True:
        success, frame = stream.read_frame()
        if success:
            faces = detector.extract_faces(frame)
            visual_score = 0.0

            for face in faces:
                x, y, w, h = face["bbox"]
                v_metrics = visual_analyzer.analyze_face(face["crop"])
                visual_score = v_metrics["fake_score"]

                # Evaluate fused score per detected face
                threat = fusion_engine.evaluate_threat(visual_score, latest_audio_score)
                
                color = (0, 0, 255) if threat["alert_level"] == "HIGH" else (0, 255, 255) if threat["alert_level"] == "MEDIUM" else (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Vis: {visual_score:.2f} | Aud: {latest_audio_score:.2f}", 
                            (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Global stream HUD
            if not faces:
                threat = fusion_engine.evaluate_threat(0.0, latest_audio_score)

            hud_color = (0, 0, 255) if threat["alert_level"] == "HIGH" else (0, 255, 255) if threat["alert_level"] == "MEDIUM" else (0, 255, 0)
            cv2.putText(frame, f"STATUS: {threat['status']}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
            cv2.putText(frame, f"COMPOSITE SCORE: {threat['composite_score']}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Enterprise Multimodal Deepfake Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    stream.stop()
    cv2.destroyAllWindows()