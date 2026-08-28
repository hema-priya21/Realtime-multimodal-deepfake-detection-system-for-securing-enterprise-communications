import cv2
import time
from buffer_manager import FrameBufferManager

if __name__ == "__main__":
    print("Starting Phase 1 Live Stream Ingestion...")
    stream = FrameBufferManager(src=0).start()
    time.sleep(1.0)

    while True:
        success, frame = stream.read_frame()
        if success:
            cv2.putText(frame, "Phase 1: Stream Active", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Live Ingest Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()