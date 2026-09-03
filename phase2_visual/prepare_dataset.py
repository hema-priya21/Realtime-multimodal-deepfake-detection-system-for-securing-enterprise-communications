import cv2
import os
import time
import urllib.request
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Project root is one folder above phase2_visual
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "ff_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "visual"

FRAMES_PER_VIDEO = 15
FACE_SIZE = 224

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}

# Face detector file will be stored inside your project
CASCADE_DIR = Path(__file__).resolve().parent / "haarcascades"
CASCADE_PATH = CASCADE_DIR / "haarcascade_frontalface_default.xml"

# Official OpenCV cascade file
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "4.x/data/haarcascades/haarcascade_frontalface_default.xml"
)


# ============================================================
# DOWNLOAD HAAR CASCADE IF MISSING
# ============================================================

def ensure_cascade():

    if CASCADE_PATH.exists():
        return

    print()
    print("Haar Cascade file not found.")
    print("Downloading face detector...")

    CASCADE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        urllib.request.urlretrieve(
            CASCADE_URL,
            str(CASCADE_PATH)
        )

    except Exception as e:

        print()
        print("ERROR: Could not download Haar Cascade.")
        print()
        print("Error:", e)
        print()
        print("Please check your internet connection and")
        print("run the program again.")
        print()

        raise SystemExit(1)

    if not CASCADE_PATH.exists():

        print("ERROR: Cascade file was not created.")
        raise SystemExit(1)

    print("Face detector downloaded successfully.")


# ============================================================
# FACE DETECTOR
# ============================================================

class FaceDetector:

    def __init__(self):

        ensure_cascade()

        self.face_cascade = cv2.CascadeClassifier(
            str(CASCADE_PATH)
        )

        if self.face_cascade.empty():

            raise RuntimeError(
                "Could not load Haar Cascade face detector."
            )

    def detect(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return faces


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    folders = [
        OUTPUT_DIR / "train" / "real",
        OUTPUT_DIR / "train" / "fake",
        OUTPUT_DIR / "val" / "real",
        OUTPUT_DIR / "val" / "fake"
    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# DETERMINE REAL / FAKE LABEL
# ============================================================

def get_label(video_path):

    # Convert complete path to lowercase
    path_string = str(video_path).lower()

    parts = [
        part.lower()
        for part in video_path.parts
    ]

    filename = video_path.name.lower()

    # --------------------------------------------------------
    # Explicit real/fake names
    # --------------------------------------------------------

    if "real" in filename:
        return "real"

    if "fake" in filename:
        return "fake"

    # --------------------------------------------------------
    # Check folder names
    # --------------------------------------------------------

    for part in parts:

        if part in {
            "real",
            "original",
            "originals",
            "original_sequences",
            "original_sequences_youtube"
        }:
            return "real"

        if part in {
            "fake",
            "fakes",
            "manipulated",
            "manipulated_sequences",
            "manipulated_sequences_youtube"
        }:
            return "fake"

    # --------------------------------------------------------
    # Check complete path for common FF++ names
    # --------------------------------------------------------

    if "original_sequences" in path_string:
        return "real"

    if "manipulated_sequences" in path_string:
        return "fake"

    return None


# ============================================================
# CHECK IF VIDEO WAS ALREADY PROCESSED
# ============================================================

def already_processed(video_path, output_dir):

    pattern = (
        f"{video_path.stem}_frame*_face*.jpg"
    )

    existing_files = list(
        output_dir.glob(pattern)
    )

    return len(existing_files) > 0


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_dir,
    detector
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return -1

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        return -1

    # --------------------------------------------------------
    # Select frames evenly throughout video
    # --------------------------------------------------------

    frame_positions = []

    for i in range(FRAMES_PER_VIDEO):

        position = int(
            i * total_frames / FRAMES_PER_VIDEO
        )

        frame_positions.append(position)

    extracted_faces = 0

    # --------------------------------------------------------
    # Process selected frames
    # --------------------------------------------------------

    for frame_number, position in enumerate(
        frame_positions
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            position
        )

        success, frame = cap.read()

        if not success:
            continue

        faces = detector.detect(frame)

        for face_number, (x, y, w, h) in enumerate(
            faces
        ):

            # ------------------------------------------------
            # Add margin around face
            # ------------------------------------------------

            margin = int(
                0.15 * max(w, h)
            )

            x1 = max(
                0,
                x - margin
            )

            y1 = max(
                0,
                y - margin
            )

            x2 = min(
                frame.shape[1],
                x + w + margin
            )

            y2 = min(
                frame.shape[0],
                y + h + margin
            )

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # Resize face
            # ------------------------------------------------

            face = cv2.resize(
                face,
                (FACE_SIZE, FACE_SIZE)
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            filename = (
                f"{video_path.stem}"
                f"_frame{frame_number:02d}"
                f"_face{face_number:02d}.jpg"
            )

            output_path = output_dir / filename

            # Don't overwrite existing file
            if output_path.exists():
                continue

            saved = cv2.imwrite(
                str(output_path),
                face
            )

            if saved:

                extracted_faces += 1

    cap.release()

    return extracted_faces


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    if seconds < 0:
        seconds = 0

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "       REAL-TIME MULTIMODAL DEEPFAKE DATASET PREPARATION"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Check input folder
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        print()
        print("ERROR: Input folder does not exist:")
        print(INPUT_DIR)
        print()

        print(
            "Make sure your videos are inside:"
        )

        print(
            "data\\ff_raw"
        )

        print()

        return

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Load face detector
    # --------------------------------------------------------

    print()
    print("Loading face detector...")

    detector = FaceDetector()

    print("Face detector loaded successfully.")

    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    print()
    print("Searching for videos...")

    video_files = []

    for file in INPUT_DIR.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in VIDEO_EXTENSIONS
        ):

            video_files.append(file)

    video_files.sort()

    total_videos = len(video_files)

    # --------------------------------------------------------
    # No videos
    # --------------------------------------------------------

    if total_videos == 0:

        print()
        print("ERROR: No video files were found.")
        print()
        print(
            f"Checked: {INPUT_DIR}"
        )
        print()

        return

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("DATASET INFORMATION")
    print("=" * 75)

    print(
        f"Input folder       : {INPUT_DIR}"
    )

    print(
        f"Total videos       : {total_videos:,}"
    )

    print(
        f"Frames per video   : {FRAMES_PER_VIDEO}"
    )

    print(
        f"Maximum frames     : "
        f"{total_videos * FRAMES_PER_VIDEO:,}"
    )

    print(
        f"Face size          : "
        f"{FACE_SIZE} x {FACE_SIZE}"
    )

    print(
        f"Output folder      : {OUTPUT_DIR}"
    )

    print("=" * 75)

    print()
    print("Starting processing...")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    start_time = time.time()

    total_faces = 0

    processed_videos = 0

    skipped_videos = 0

    failed_videos = 0

    unknown_label_videos = 0

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for index, video_path in enumerate(
        video_files,
        start=1
    ):

        video_start_time = time.time()

        percentage = (
            index
            / total_videos
            * 100
        )

        # ----------------------------------------------------
        # Get label
        # ----------------------------------------------------

        label = get_label(video_path)

        if label is None:

            unknown_label_videos += 1

            print()
            print(
                f"[{index}/{total_videos}] "
                f"{percentage:6.2f}% | SKIPPED"
            )

            print(
                f"Video: {video_path.name}"
            )

            print(
                "Reason: Could not determine REAL/FAKE label"
            )

            continue

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        output_dir = (
            OUTPUT_DIR
            / "train"
            / label
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Check already processed
        # ----------------------------------------------------

        if already_processed(
            video_path,
            output_dir
        ):

            skipped_videos += 1

            elapsed = (
                time.time()
                - start_time
            )

            average_time = (
                elapsed
                / index
            )

            remaining_videos = (
                total_videos
                - index
            )

            eta = (
                average_time
                * remaining_videos
            )

            print()
            print(
                f"[{index}/{total_videos}] "
                f"{percentage:6.2f}% | "
                f"ALREADY PROCESSED"
            )

            print(
                f"Video: {video_path.name}"
            )

            print(
                f"Label: {label.upper()}"
            )

            print(
                f"ETA: {format_time(eta)}"
            )

            continue

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        try:

            faces = process_video(
                video_path,
                output_dir,
                detector
            )

        except Exception as e:

            failed_videos += 1

            print()
            print(
                f"[{index}/{total_videos}] "
                f"{percentage:6.2f}% | ERROR"
            )

            print(
                f"Video: {video_path.name}"
            )

            print(
                f"Error: {e}"
            )

            continue

        # ----------------------------------------------------
        # Failed video
        # ----------------------------------------------------

        if faces == -1:

            failed_videos += 1

            print()
            print(
                f"[{index}/{total_videos}] "
                f"{percentage:6.2f}% | "
                f"FAILED"
            )

            print(
                f"Video: {video_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        processed_videos += 1

        total_faces += faces

        video_time = (
            time.time()
            - video_start_time
        )

        elapsed = (
            time.time()
            - start_time
        )

        average_time = (
            elapsed
            / index
        )

        remaining_videos = (
            total_videos
            - index
        )

        estimated_remaining = (
            average_time
            * remaining_videos
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print()
        print("-" * 75)

        print(
            f"VIDEO {index:,} / "
            f"{total_videos:,}"
        )

        print(
            f"Progress          : "
            f"{percentage:6.2f}%"
        )

        print(
            f"Video             : "
            f"{video_path.name}"
        )

        print(
            f"Label             : "
            f"{label.upper()}"
        )

        print(
            f"Faces extracted   : "
            f"{faces}"
        )

        print(
            f"Total faces       : "
            f"{total_faces:,}"
        )

        print(
            f"Video time        : "
            f"{video_time:.2f} seconds"
        )

        print(
            f"Average/video     : "
            f"{average_time:.2f} seconds"
        )

        print(
            f"Remaining videos  : "
            f"{remaining_videos:,}"
        )

        print(
            f"Estimated time    : "
            f"{format_time(estimated_remaining)}"
        )

        print("-" * 75)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )

    print()
    print()
    print("=" * 75)
    print(
        "                 PROCESSING COMPLETE"
    )
    print("=" * 75)

    print(
        f"Videos found          : "
        f"{total_videos:,}"
    )

    print(
        f"Videos processed      : "
        f"{processed_videos:,}"
    )

    print(
        f"Videos skipped        : "
        f"{skipped_videos:,}"
    )

    print(
        f"Videos failed         : "
        f"{failed_videos:,}"
    )

    print(
        f"Unknown labels        : "
        f"{unknown_label_videos:,}"
    )

    print(
        f"Faces extracted       : "
        f"{total_faces:,}"
    )

    print(
        f"Total processing time : "
        f"{format_time(total_time)}"
    )

    if total_videos > 0:

        print(
            f"Average/video         : "
            f"{total_time / total_videos:.2f} seconds"
        )

    print("=" * 75)

    print()
    print(
        "Face extraction finished."
    )

    print(
        f"Images saved to: {OUTPUT_DIR}"
    )

    print()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()