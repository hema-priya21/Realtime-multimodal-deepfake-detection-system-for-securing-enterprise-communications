import numpy as np

class AudioAnalyzer:
    """Analyzes audio signal chunks for synthetic voice artifacts using spectral metrics."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def analyze_chunk(self, audio_data):
        """Calculates baseline spectral flatline and frequency anomaly score (0.0 = Real, 1.0 = Fake)."""
        if len(audio_data) == 0:
            return {"fake_score": 0.0, "spectral_flatness": 0.0}

        signal = np.array(audio_data, dtype=np.float32)

        # RMS volume filter to ignore silence and background static
        rms = np.sqrt(np.mean(signal**2))
        if rms < 0.01:
            return {"fake_score": 0.0, "spectral_flatness": 0.0}

        # Fast Fourier Transform (FFT)
        fft_vals = np.abs(np.fft.rfft(signal))
        
        # Spectral flatness calculation
        geometric_mean = np.exp(np.mean(np.log(fft_vals + 1e-12)))
        arithmetic_mean = np.mean(fft_vals) + 1e-12
        flatness = geometric_mean / arithmetic_mean

        # Calibrated heuristic score mapping
        fake_score = max(0.0, min(1.0, float((flatness - 0.2) * 2.0)))

        return {
            "fake_score": round(fake_score, 4),
            "spectral_flatness": round(float(flatness), 4)
        }