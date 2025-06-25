import os
import glob
import cv2
import numpy as np
from feature_extraction import run_yolo_on_frames, save_detections_json
import concurrent.futures
import argparse

# Helper to draw bounding boxes and class labels on a frame
def draw_detections(frame, detections):
    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        conf = det['conf']
        color = (0, 255, 0)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return frame

EXTRACTED_FRAMES_DIR = "extracted_frames"
DETECTIONS_DIR = "detections"
YOLO_MODEL_PATH = "yolo11n.pt"  # or your preferred YOLOv11 model
DEVICE = "cuda"  # or "cpu"

MAX_WORKERS = 4  # Adjust based on your CPU/GPU capability

os.makedirs(DETECTIONS_DIR, exist_ok=True)


def process_video(video_id):
    video_dir = os.path.join(EXTRACTED_FRAMES_DIR, video_id)
    out_dir = os.path.join(DETECTIONS_DIR, video_id)
    vis_dir = os.path.join('detections_vis', video_id)
    if os.path.exists(out_dir):
        print(f"Skipping {video_id}, detections already exist.")
        return
    print(f"Processing video {video_id}...")
    frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")) + glob.glob(os.path.join(video_dir, "*.png")))

    valid_frames = []
    valid_frame_files = []
    skipped_files = []
    for idx, f in enumerate(frame_files):
        # Debug: print info for the first 3 PNGs in 00005
        if video_id == '00005' and idx < 3 and f.lower().endswith('.png'):
            abs_path = os.path.abspath(f)
            print(f"[DEBUG] Checking file: {f}")
            print(f"[DEBUG] Absolute path: {abs_path}")
            print(f"[DEBUG] Exists: {os.path.exists(abs_path)}")
            test_img = cv2.imread(abs_path)
            print(f"[DEBUG] cv2.imread result: {type(test_img)}, shape: {getattr(test_img, 'shape', None)}")
        try:
            img = cv2.imread(f)
            if img is not None and isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3:
                valid_frames.append(img)
                valid_frame_files.append(f)
            else:
                print(f"[WARN] {f} could not be read properly by cv2 or is not a valid 3-channel image and will be skipped.")
                skipped_files.append(f)
        except Exception as e:
            print(f"[ERROR] Exception reading {f}: {e}")
            skipped_files.append(f)

    if skipped_files:
        print(f"[WARN] {len(skipped_files)} files could not be read and were skipped in {video_id}:")
        for fname in skipped_files:
            print(f"  - {fname}")
        # Log skipped filenames to detections_vis/<video_id>/skipped_files.txt
        os.makedirs(vis_dir, exist_ok=True)
        skipped_log = os.path.join(vis_dir, 'skipped_files.txt')
        with open(skipped_log, 'w') as f:
            for fname in skipped_files:
                f.write(fname + '\n')
    if not valid_frames:
        print(f"[ERROR] No valid frames found for {video_id}. Skipping video.")
        return

    # Frame integrity checks and logging
    bad_frames = []
    for i, (img, fname) in enumerate(zip(valid_frames, valid_frame_files)):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
            print(f"[ERROR] Frame {i} ({fname}) is invalid: type={type(img)}, dtype={getattr(img, 'dtype', None)}, shape={getattr(img, 'shape', None)}")
            bad_frames.append(fname)
        elif img.shape[2] != 3:
            print(f"[ERROR] Frame {i} ({fname}) does not have 3 channels, shape={img.shape}")
            bad_frames.append(fname)
        elif np.all(img == 0):
            print(f"[WARN] Frame {i} ({fname}) is all zeros (black image)")
    # Log bad frames to detections_vis/<video_id>/bad_frames.txt
    if bad_frames:
        bad_frames_log = os.path.join(vis_dir, 'bad_frames.txt')
        with open(bad_frames_log, 'w') as f:
            for fname in bad_frames:
                f.write(fname + '\n')
            bad_frames.append((i, fname))
    if bad_frames:
        os.makedirs(vis_dir, exist_ok=True)
        bad_log = os.path.join(vis_dir, 'bad_input_frames.txt')
        with open(bad_log, 'w') as f:
            for i, fname in bad_frames:
                f.write(f"{i}: {fname}\n")
        print(f"[ERROR] {len(bad_frames)} bad input frames found for {video_id}. See {bad_log}")
        # Optionally, skip bad frames for YOLO
        valid_frames = [img for i, img in enumerate(valid_frames) if i not in [idx for idx, _ in bad_frames]]
        valid_frame_files = [fname for i, fname in enumerate(valid_frame_files) if i not in [idx for idx, _ in bad_frames]]
        if not valid_frames:
            print(f"[ERROR] No valid frames remain after removing bad frames for {video_id}. Skipping video.")
            return

    detections = run_yolo_on_frames(valid_frames, model_path=YOLO_MODEL_PATH, device=DEVICE)
    save_detections_json(detections, out_dir)
    print(f"Detections for {video_id} saved to {out_dir}")

    # Visualization: save frames with boxes
    os.makedirs(vis_dir, exist_ok=True)
    failed_writes = []
    for idx, (frame, frame_dets) in enumerate(zip(valid_frames, detections)):
        vis_frame = draw_detections(frame.copy(), frame_dets)
        vis_path = os.path.join(vis_dir, f"frame_{idx:05d}.png")
        success = cv2.imwrite(vis_path, vis_frame)
        if not success:
            print(f"[ERROR] Failed to write visualization frame {vis_path}")
            failed_writes.append(vis_path)
    if failed_writes:
        failed_log = os.path.join(vis_dir, 'failed_writes.txt')
        with open(failed_log, 'w') as f:
            for fname in failed_writes:
                f.write(fname + '\n')
        print(f"[ERROR] {len(failed_writes)} visualization frames failed to write for {video_id}. See {failed_log}")
    print(f"Visualization for {video_id} saved to {vis_dir}")

    # Combine visualized frames into a video
    vis_frame_files = sorted(glob.glob(os.path.join(vis_dir, "frame_*.png")))
    valid_video_frames = []
    skipped_vis_pngs = []
    for f in vis_frame_files:
        img = cv2.imread(f)
        if img is not None:
            valid_video_frames.append(img)
        else:
            print(f"[WARN] Could not read visualization PNG {f}, skipping from video.")
            skipped_vis_pngs.append(f)
    if valid_video_frames:
        height, width = valid_video_frames[0].shape[:2]
        video_path = os.path.join(vis_dir, "detections.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for img in valid_video_frames:
            out_video.write(img)
        out_video.release()
        print(f"Detection video for {video_id} saved to {video_path}")
        # Optionally log skipped visualization PNGs
        if skipped_vis_pngs:
            skipped_log = os.path.join(vis_dir, 'skipped_visualization_pngs.txt')
            with open(skipped_log, 'w') as f:
                for fname in skipped_vis_pngs:
                    f.write(fname + '\n')
        # Delete .png visualization frames to save disk space
        for f in vis_frame_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"[WARN] Could not delete {f}: {e}")
    else:
        print(f"[ERROR] No valid visualization frames found for {video_id}. Skipping video creation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch YOLOv11 detection on extracted video frames.")
    parser.add_argument('--num_videos', type=int, default=None, help='Number of videos to process (default: all)')
    args = parser.parse_args()

    video_ids = [d for d in sorted(os.listdir(EXTRACTED_FRAMES_DIR)) if os.path.isdir(os.path.join(EXTRACTED_FRAMES_DIR, d))]
    # Filter out videos where detections already exist
    video_ids_to_process = [vid for vid in video_ids if not os.path.exists(os.path.join(DETECTIONS_DIR, vid))]
    if args.num_videos is not None:
        video_ids_to_process = video_ids_to_process[:args.num_videos]
        print(f"Processing only the first {args.num_videos} videos.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_video, video_ids_to_process))
