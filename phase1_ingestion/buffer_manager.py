import cv2
import queue
import threading
import time

class FrameBufferManager:
    """Thread-safe frame buffer for real-time video stream ingestion."""
    def __init__(self, src=0, max_queue_size=30):
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.01)

    def read_frame(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()