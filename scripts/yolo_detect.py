"""
yolo_detect.py

Script to run batch YOLOv11 detection on all frames in a directory, saving detection results and visualizations.
Uses core logic from src/yolo_utils.py.

Usage: python scripts/yolo_detect.py --frames_dir <frames> --output_dir <detections> --num_videos <num_videos>
"""

import argparse
from src.yolo_utils import load_yolo_model, run_yolo_on_image, parse_yolo_results

from tqdm import tqdm
import os
import cv2
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_videos', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_yolo_model(args.model_path, version='yolov11')
    video_dirs = [d for d in os.listdir(args.frames_dir) if os.path.isdir(os.path.join(args.frames_dir, d))]
    processed_videos = 0
    for video_dir in tqdm(video_dirs, desc='Videos'):
        video_path = os.path.join(args.frames_dir, video_dir)
        output_path = os.path.join(args.output_dir, video_dir)
        if os.path.exists(output_path) and len(os.listdir(output_path)) == len(os.listdir(video_path)):
            continue
        frame_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in tqdm(frame_files, desc='Frames'):
            img_path = os.path.join(video_path, fname)
            img = cv2.imread(img_path)
            results = run_yolo_on_image(model, img)
            parsed = parse_yolo_results(results)
            # Save detections as JSON
            out_path = os.path.join(output_path, fname.rsplit('.', 1)[0] + '.json')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(parsed, f)
            # Optional: save visualization
            if args.visualize:
                vis_img = img.copy()
                for det in parsed:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    conf = det['conf']
                    label = det['class_name']
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(vis_img, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                vis_path = os.path.join(args.output_dir + '_vis', video_dir, fname)
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                cv2.imwrite(vis_path, vis_img)
        processed_videos += 1
        if processed_videos >= args.num_videos:
            break

if __name__ == "__main__":
    main()
