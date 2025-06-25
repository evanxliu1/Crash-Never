import os
import numpy as np
import json
from tqdm import tqdm

# Classes to include in YOLO stats
YOLO_CLASSES = [
    'car', 'person', 'truck', 'bus', 'traffic light',
    'motorcycle', 'bicycle', 'fire hydrant', 'stop sign'
]

# Helper to extract YOLO stats for selected classes

def extract_yolo_stats(detections, classes=YOLO_CLASSES):
    stats = np.zeros(len(classes), dtype=np.float32)
    for det in detections:
        if det.get('class_name') in classes:
            idx = classes.index(det['class_name'])
            stats[idx] += 1  # count instances per class
    return stats

def extract_flow_stats(flow_stat):
    # Assume flow_stat is a dict with keys: mag_mean, mag_std, ang_mean, ang_std
    return np.array([
        flow_stat.get('mag_mean', 0.0),
        flow_stat.get('mag_std', 0.0),
        flow_stat.get('ang_mean', 0.0),
        flow_stat.get('ang_std', 0.0)
    ], dtype=np.float32)

def combine_features(
    cnn_root, detections_root, flow_root, output_root, max_videos=76
):
    os.makedirs(output_root, exist_ok=True)
    video_ids = sorted([f[:-4] for f in os.listdir(cnn_root) if f.endswith('.npz')])[:max_videos]
    for vid in tqdm(video_ids, desc='Videos'):
        # Load CNN features
        cnn_path = os.path.join(cnn_root, f"{vid}.npz")
        cnn_features = np.load(cnn_path)['features']  # (num_frames, 512)
        # Load YOLO detections
        det_dir = os.path.join(detections_root, vid)
        det_files = sorted([f for f in os.listdir(det_dir) if f.endswith('.json')])
        # Load optical flow stats
        flow_path = os.path.join(flow_root, vid, "flow_stats.json")
        if not os.path.exists(flow_path):
            print(f"Warning: missing flow stats for video {vid}, skipping.")
            continue
        with open(flow_path, 'r') as f:
            flow_stats = json.load(f)  # list of dicts, one per frame
        # Align lengths (robust to missing/extra frames)
        min_len = min(len(cnn_features), len(det_files), len(flow_stats))
        cnn_features = cnn_features[:min_len]
        det_files = det_files[:min_len]
        flow_stats = flow_stats[:min_len]
        # Build per-frame feature vectors
        combined = []
        for i in range(min_len):
            # YOLO stats
            with open(os.path.join(det_dir, det_files[i]), 'r') as f:
                dets = json.load(f)
            yolo_vec = extract_yolo_stats(dets)
            flow_vec = extract_flow_stats(flow_stats[i])
            feat = np.concatenate([cnn_features[i], yolo_vec, flow_vec])
            combined.append(feat)
        combined = np.stack(combined, axis=0)  # (num_frames, 512+9+4)
        out_path = os.path.join(output_root, f"{vid}_combined.npy")
        np.save(out_path, combined)
        print(f"Saved {combined.shape} features for video {vid} to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine CNN, YOLO, and flow features per frame.")
    parser.add_argument('--cnn_root', type=str, required=True, help='Directory with .npz CNN features')
    parser.add_argument('--detections_root', type=str, required=True, help='Root directory of YOLO detection JSONs')
    parser.add_argument('--flow_root', type=str, required=True, help='Root directory of optical flow stats (json)')
    parser.add_argument('--output_root', type=str, required=True, help='Output directory for combined features')
    parser.add_argument('--max_videos', type=int, default=76, help='Number of videos to process')
    args = parser.parse_args()
    combine_features(args.cnn_root, args.detections_root, args.flow_root, args.output_root, args.max_videos)
