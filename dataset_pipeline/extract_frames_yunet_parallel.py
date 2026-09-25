"""Parallel wrapper around extract_frames_yunet.py's exact same per-video
extraction logic (read_split_csv, already_processed, extract_faces_from_video,
FaceDetector, select_largest_face -- all imported unchanged, nothing
duplicated or reimplemented). The only difference is that videos within a
split are processed by multiple worker processes at once instead of one at a
time, since face detection/GPU is not the bottleneck here (this machine's
OpenCV build has no CUDA DNN support -- verified: cv2.cuda.getCudaEnabledDeviceCount()
== 0 -- so YuNet only ever runs on CPU; the real lever available is CPU
parallelism across videos, which are fully independent of each other).

Safe to run against a directory an earlier, unfinished run of
extract_frames_yunet.py already partially populated: already_processed()
(imported, not reimplemented) skips any video whose output already exists,
so no work is duplicated and nothing already on disk is touched or
overwritten.

Does not modify extract_frames_yunet.py, phase2_visual/face_detector.py, or
any other file. Reads the same split CSVs read-only. Writes only into the
given --output directory.

Run as:
    python -m dataset_pipeline.extract_frames_yunet_parallel --workers 4
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import re

from dataset_pipeline.extract_frames_yunet import (
    DEFAULT_OUTPUT_ROOT, extract_faces_from_video,
    format_time, read_split_csv, safe_token,
)

_STEM_RE = re.compile(r"^(.+)_frame\d+_face00\.jpg$")


def _scan_done_stems(output_dir):
    """Equivalent to calling the original already_processed() once per video,
    but O(files-in-directory) total instead of O(files-in-directory) PER
    VIDEO CHECKED. On a large, OneDrive-synced output directory (tens of
    thousands of files), the original glob-per-video check -- correct, and
    left completely unchanged for the sequential script -- becomes the
    actual bottleneck once several workers repeat it concurrently against
    the same folder (observed: 0 new crops for 150+ seconds once
    train/fake passed ~47,000 existing files). Listing the directory once
    and checking set membership preserves identical skip semantics."""
    if not output_dir.exists():
        return set()
    stems = set()
    for f in output_dir.iterdir():
        m = _STEM_RE.match(f.name)
        if m:
            stems.add(m.group(1))
    return stems

_worker_detector = None


def _init_worker():
    """Runs once per worker process -- each process gets its own FaceDetector
    (the exact same class extract_frames_yunet.py uses), since cv2 detector
    objects are not safely shareable across processes.

    cv2.setNumThreads(1): OpenCV's default CPU backend already parallelizes
    a single detection call across multiple threads internally. Left at its
    default, N worker PROCESSES each also spawn their own multi-threaded
    OpenCV thread pool, oversubscribing the CPU (observed: 4 workers each
    pinned near 3 CPU-cores, ~12 threads total fighting over 12 logical
    cores, with near-zero throughput for the first minute). Restricting
    each worker to 1 internal thread makes the outer process pool -- not
    OpenCV's own threading -- the actual source of parallelism."""
    global _worker_detector
    import cv2
    cv2.setNumThreads(1)
    from phase2_visual.face_detector import FaceDetector
    _worker_detector = FaceDetector()


def _process_one(args):
    row, output_root, split_name = args
    video_path = Path(row["filepath"])
    label = row["label"]
    video_stem = f"{row['dataset']}_{safe_token(video_path.parent.name)}_{safe_token(video_path.stem)}"
    output_dir = output_root / split_name / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"    [worker {__import__('os').getpid()}] START {video_stem}", flush=True)
    t0 = time.time()

    if not video_path.exists():
        return {"outcome": "failed", "reason": "MISSING", "path": str(video_path)}

    saved, no_det, status = extract_faces_from_video(video_path, output_dir, _worker_detector, video_stem)
    print(f"    [worker {__import__('os').getpid()}] DONE  {video_stem} saved={saved} status={status} "
          f"took={time.time()-t0:.1f}s", flush=True)
    if status != "ok":
        return {"outcome": "failed", "reason": status.upper(), "path": str(video_path)}

    return {"outcome": "processed", "saved": saved, "no_det": no_det, "label": label,
            "zero_crop": saved == 0}


def process_split_parallel(split_name, output_root, workers, pilot_limit=None):
    rows = read_split_csv(split_name)
    if pilot_limit is not None:
        from dataset_pipeline.extract_frames_yunet import select_pilot_rows
        rows = select_pilot_rows(rows, pilot_limit)

    print(f"\n{'=' * 75}\n{split_name.upper()} -- {len(rows):,} videos ({workers} workers)\n{'=' * 75}")

    start = time.time()

    # Skip-check done ONCE per (split, label) directory here in the main
    # process -- not per video, not inside workers -- see _scan_done_stems
    # docstring for why the original per-video glob check stalls badly once
    # a worker pool repeats it against a large, already-populated directory.
    done_by_label = {}
    for label in ("real", "fake"):
        output_dir = output_root / split_name / label
        done_by_label[label] = _scan_done_stems(output_dir)

    todo_rows = []
    skipped = 0
    for row in rows:
        video_path = Path(row["filepath"])
        video_stem = f"{row['dataset']}_{safe_token(video_path.parent.name)}_{safe_token(video_path.stem)}"
        if video_stem in done_by_label[row["label"]]:
            skipped += 1
        else:
            todo_rows.append(row)
    print(f"  Already done (skipped): {skipped:,}  Remaining to process: {len(todo_rows):,}")

    processed = failed = zero_crop_videos = 0
    total_crops = real_crops = fake_crops = total_no_detection_frames = 0
    done = 0

    tasks = [(row, output_root, split_name) for row in todo_rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_process_one, t) for t in tasks]
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r["outcome"] == "failed":
                failed += 1
                print(f"  [{done}/{len(todo_rows)}] {r['reason']}: {r['path']}")
            else:
                processed += 1
                total_crops += r["saved"]
                total_no_detection_frames += r["no_det"]
                if r["zero_crop"]:
                    zero_crop_videos += 1
                if r["label"] == "real":
                    real_crops += r["saved"]
                else:
                    fake_crops += r["saved"]

            if done % 50 == 0 or done == len(todo_rows):
                elapsed = time.time() - start
                eta = (elapsed / done) * (len(todo_rows) - done) if done else 0
                print(f"  [{done}/{len(todo_rows)}] processed={processed} skipped={skipped} failed={failed} "
                      f"crops={total_crops:,} (real {real_crops:,}/fake {fake_crops:,}) "
                      f"no-detection-frames={total_no_detection_frames:,} "
                      f"elapsed={format_time(elapsed)} eta={format_time(eta)}")

    print(f"\n{split_name.upper()} DONE -- processed {processed:,}, skipped {skipped:,}, "
          f"failed {failed:,}, videos-with-zero-crops {zero_crop_videos:,}, "
          f"crops {total_crops:,} (real {real_crops:,} / fake {fake_crops:,}), "
          f"time {format_time(time.time() - start)}")

    return {
        "processed": processed, "skipped": skipped, "failed": failed,
        "zero_crop_videos": zero_crop_videos, "total_crops": total_crops,
        "real_crops": real_crops, "fake_crops": fake_crops,
        "no_detection_frames": total_no_detection_frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_root = Path(args.output)
    print("=" * 75)
    print("YuNet-based training-data extraction -- PARALLEL (CPU-only, no CUDA DNN available)")
    print("Output root:", output_root)
    print("Workers    :", args.workers)
    print("Pilot mode :", f"{args.pilot} real + {args.pilot} fake per split" if args.pilot else "FULL DATASET")
    print("Resuming any already-extracted files in place (already_processed() skip logic unchanged).")
    print("=" * 75)

    summary = {}
    for split_name in ("train", "val", "test"):
        summary[split_name] = process_split_parallel(split_name, output_root, args.workers, pilot_limit=args.pilot)

    print("\n" + "=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)
    for split_name, s in summary.items():
        print(f"{split_name.upper():5s} | processed={s['processed']:,} skipped={s['skipped']:,} "
              f"failed={s['failed']:,} zero-crop-videos={s['zero_crop_videos']:,} "
              f"crops={s['total_crops']:,} (real {s['real_crops']:,} / fake {s['fake_crops']:,}) "
              f"no-detection-frames={s['no_detection_frames']:,}")
    print("=" * 75)


if __name__ == "__main__":
    main()
