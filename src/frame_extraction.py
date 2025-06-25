"""
frame_extraction.py

Core functions for extracting frames from videos (e.g., extracting middle 50%).
Intended to be imported by frame extraction scripts.
"""

import cv2
import os

def extract_middle_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = int(0.25 * total_frames)
    end_idx = int(0.75 * total_frames)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    idx = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i < start_idx or i >= end_idx:
            continue
        out_path = os.path.join(output_dir, f'{basename}_frame_{idx:05d}.png')
        cv2.imwrite(out_path, frame)
        idx += 1
    cap.release()
