import os
import cv2
from tqdm import tqdm

def resize_and_save_frames(frames_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for video_id in os.listdir(frames_dir):
        video_path = os.path.join(frames_dir, video_id)
        if not os.path.isdir(video_path):
            continue
        out_video_path = os.path.join(output_dir, video_id)
        os.makedirs(out_video_path, exist_ok=True)
        frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
        for fname in tqdm(frame_files, desc=f"{video_id}"):
            in_path = os.path.join(video_path, fname)
            out_path = os.path.join(out_video_path, fname)
            img = cv2.imread(in_path)
            if img is None:
                print(f"Warning: could not read {in_path}")
                continue
            resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_path, resized)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resize frames to 224x224 for CNN input.")
    parser.add_argument('--frames_dir', type=str, required=True, help='Root directory of original frames')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for resized frames')
    args = parser.parse_args()
    resize_and_save_frames(args.frames_dir, args.output_dir)
