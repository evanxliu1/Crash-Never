import os
import numpy as np
from tqdm import tqdm

def sliding_window_aggregate(
    combined_root, output_root, window_size=30, stride=6, max_videos=76
):
    os.makedirs(output_root, exist_ok=True)
    video_files = sorted([f for f in os.listdir(combined_root) if f.endswith('_combined.npy')])[:max_videos]
    for fname in tqdm(video_files, desc='Videos'):
        vid = fname.split('_')[0]
        arr = np.load(os.path.join(combined_root, fname))  # (num_frames, feature_dim)
        windows = []
        for start in range(0, arr.shape[0] - window_size + 1, stride):
            window = arr[start:start+window_size]  # (window_size, feature_dim)
            windows.append(window)
        windows = np.stack(windows, axis=0) if windows else np.zeros((0, window_size, arr.shape[1]), dtype=arr.dtype)
        out_path = os.path.join(output_root, f"{vid}_windowed.npy")
        np.save(out_path, windows)
        print(f"Saved {windows.shape} sliding windows for video {vid} to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate per-frame features into sliding windows.")
    parser.add_argument('--combined_root', type=str, required=True, help='Directory with per-frame combined .npy files')
    parser.add_argument('--output_root', type=str, required=True, help='Output directory for windowed features')
    parser.add_argument('--window_size', type=int, default=30, help='Sliding window size (frames)')
    parser.add_argument('--stride', type=int, default=6, help='Sliding window stride (frames)')
    parser.add_argument('--max_videos', type=int, default=76, help='Number of videos to process')
    args = parser.parse_args()
    sliding_window_aggregate(args.combined_root, args.output_root, args.window_size, args.stride, args.max_videos)
