import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

# Directory containing the input .mp4 videos
TRAIN_DIR = 'train'
# Directory where extracted frames will be saved
FRAMES_DIR = 'R:/extracted_frames'
# Target frame extraction rate (frames per second)
FPS = 30  # Target FPS for extraction

# Ensure the output directory for frames exists
os.makedirs(FRAMES_DIR, exist_ok=True)

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract frames from videos in batches.')
parser.add_argument('--batch_size', type=int, default=8, help='Number of videos to process in this batch')
args = parser.parse_args()

# List all .mp4 video files in the training directory that have NOT been extracted yet (limit to batch_size)
all_videos = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.mp4')]
unprocessed_videos = []
SSD_FRAMES_DIR = 'extracted_frames'
for f in all_videos:
    vid_id = os.path.splitext(f)[0]
    ssd_out = os.path.join(SSD_FRAMES_DIR, vid_id)
    if not (os.path.exists(ssd_out) and os.path.isdir(ssd_out) and len(os.listdir(ssd_out)) > 0):
        unprocessed_videos.append(f)
video_files = unprocessed_videos[:args.batch_size]

# Function to extract the middle 50% of frames from a single video and save as PNG
# - Only runs if the output frame folder is empty (prevents duplicate work)
# - Calculates the total number of frames and determines the middle 50% range
# - Samples frames at the specified FPS and saves them as PNGs
# - Skips videos that have already been processed
# - Returns (video_id, None) on success, (video_id, error) on failure

def process_video(vid_file):
    vid_id = os.path.splitext(vid_file)[0]
    vid_path = os.path.join(TRAIN_DIR, vid_file)
    frames_out = os.path.join(FRAMES_DIR, vid_id)
    # SSD output directory (relative to project root)
    SSD_FRAMES_DIR = 'extracted_frames'
    ssd_out = os.path.join(SSD_FRAMES_DIR, vid_id)
    # Check if the SSD output folder exists and is non-empty
    if os.path.exists(ssd_out) and os.path.isdir(ssd_out) and len(os.listdir(ssd_out)) > 0:
        print(f"[SKIP] Frames for {vid_id} already exist in SSD. Skipping extraction.")
        return (vid_id, None)
    os.makedirs(frames_out, exist_ok=True)
    try:
        # Only process if frames haven't already been extracted for this video in RAM disk
        if not os.listdir(frames_out):
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {vid_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            # Calculate how often to sample frames to match the target FPS
            frame_interval = int(round(orig_fps / FPS)) if orig_fps > FPS else 1
            # Only keep the middle 50% of frames
            start_frame = int(total_frames * 0.25)
            end_frame = int(total_frames * 0.75)
            frame_idx = 0
            saved_idx = 0
            # Loop through frames, saving only those in the middle 50% at the desired interval
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx >= start_frame and frame_idx < end_frame:
                    if frame_idx % frame_interval == 0:
                        out_path = os.path.join(frames_out, f"frame_{saved_idx:05d}.png")
                        cv2.imwrite(out_path, frame)
                        saved_idx += 1
                frame_idx += 1
            cap.release()
        return (vid_id, None)
    except Exception as e:
        return (vid_id, str(e))

# Main function to process all videos in parallel for speed
# - Uses up to 8 parallel processes (or as many CPUs as available)
# - Submits each video for parallel processing
# - Displays progress bar and collects any failures

def main():
    failures = []
    max_workers = min(8, os.cpu_count() or 1)  # Use up to 8 processes or CPU count
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, vid_file): vid_file for vid_file in video_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc='Extracting Frames'):
            vid_id, err = f.result()
            if err:
                failures.append((vid_id, err))
    print('Frame extraction complete for all videos.')
    if failures:
        print(f"\n{len(failures)} videos failed to extract frames:")
        for vid_id, err in failures:
            print(f"  {vid_id}: {err}")

if __name__ == '__main__':
    main()
