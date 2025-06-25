"""
extract_frames.py

Script to extract the middle 50% of frames from each video in a directory using src/frame_extraction.py.

Usage: python scripts/extract_frames.py --video_dir <videos> --output_dir <frames>
"""

import argparse
from src.frame_extraction import extract_middle_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    import os
    from tqdm import tqdm
    video_files = [f for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    for vid in tqdm(video_files, desc='Videos'):
        video_path = os.path.join(args.video_dir, vid)
        extract_middle_frames(video_path, args.output_dir)

if __name__ == "__main__":
    main()
