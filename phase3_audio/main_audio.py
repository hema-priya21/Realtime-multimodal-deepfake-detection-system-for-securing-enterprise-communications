import sounddevice as sd
import numpy as np
import time
from audio_analyzer import AudioAnalyzer

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # half-second audio chunks

if __name__ == "__main__":
    analyzer = AudioAnalyzer(sample_rate=SAMPLE_RATE)
    print("Phase 3 Audio Engine Active... Listening to microphone input (Press Ctrl+C to exit)")

    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        while True:
            # Capture real-time microphone stream chunk
            audio_chunk = sd.rec(chunk_size, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            flat_audio = audio_chunk.flatten()
            metrics = analyzer.analyze_chunk(flat_audio)

            status = "SUSPICIOUS (AI Voice)" if metrics["fake_score"] > 0.6 else "CLEAN (Real Voice)"
            print(f"[{status}] Audio Fake Score: {metrics['fake_score']} | Flatness: {metrics['spectral_flatness']}")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping Audio Engine...")