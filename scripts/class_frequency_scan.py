import os
import json
from collections import Counter, defaultdict
from tqdm import tqdm

def scan_detection_classes(detections_root, output_path="class_frequency_report.txt"):
    class_counter = Counter()
    per_video_counter = defaultdict(Counter)
    video_dirs = [d for d in os.listdir(detections_root) if os.path.isdir(os.path.join(detections_root, d))]
    for video_dir in tqdm(video_dirs, desc="Videos"):
        video_path = os.path.join(detections_root, video_dir)
        frame_files = [f for f in os.listdir(video_path) if f.endswith('.json')]
        for fname in tqdm(frame_files, desc=f"{video_dir}", leave=False):
            fpath = os.path.join(video_path, fname)
            with open(fpath, 'r') as f:
                try:
                    dets = json.load(f)
                except Exception as e:
                    print(f"Could not load {fpath}: {e}")
                    continue
                for det in dets:
                    class_name = det.get('class_name', 'unknown')
                    class_counter[class_name] += 1
                    per_video_counter[video_dir][class_name] += 1
    # Write results to txt file
    with open(output_path, 'w') as out:
        out.write("Overall class frequency:\n")
        for cls, count in class_counter.most_common():
            out.write(f"{cls}: {count}\n")
        out.write("\nPer-video class frequency (top 5 per video):\n")
        for vid, counter in per_video_counter.items():
            out.write(f"{vid}:\n")
            for cls, count in counter.most_common(5):
                out.write(f"  {cls}: {count}\n")
    print(f"Results saved to {output_path}")
    return class_counter, per_video_counter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan YOLO detection JSONs for class frequency.")
    parser.add_argument('--detections_root', type=str, required=True, help='Root directory of YOLO detection JSONs')
    parser.add_argument('--output_path', type=str, default="class_frequency_report.txt", help='Path to save the class frequency report')
    args = parser.parse_args()
    scan_detection_classes(args.detections_root, args.output_path)
