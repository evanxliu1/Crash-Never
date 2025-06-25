import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from cnn_feature_extractor import CNNFeatureExtractor

def precompute_cnn_features(frames_root, output_root, max_videos=76, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    os.makedirs(output_root, exist_ok=True)
    video_dirs = sorted([d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))])[:max_videos]
    model = CNNFeatureExtractor(pretrained=True).to(device)
    model.eval()
    with torch.no_grad():
        for video_id in tqdm(video_dirs, desc='Videos'):
            video_path = os.path.join(frames_root, video_id)
            frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
            features = []
            batch_imgs = []
            batch_idx = []
            frame_iter = tqdm(enumerate(frame_files), total=len(frame_files), desc=f"Frames ({video_id})", leave=False)
            for idx, fname in frame_iter:
                img_path = os.path.join(video_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: could not read {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)  # (1,3,224,224)
                batch_imgs.append(img)
                batch_idx.append(idx)
                if len(batch_imgs) == batch_size or idx == len(frame_files)-1:
                    batch = torch.cat(batch_imgs, dim=0).to(device)
                    feats = model(batch).cpu().numpy()  # (batch, 512)
                    features.append(feats)
                    batch_imgs = []
                    batch_idx = []
            if features:
                features = np.concatenate(features, axis=0)  # (num_frames, 512)
            else:
                features = np.zeros((0,512), dtype=np.float32)
            out_path = os.path.join(output_root, f"{video_id}.npz")
            np.savez_compressed(out_path, features=features)
            print(f"Saved {features.shape[0]} features for video {video_id} to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute CNN features for all frames (224x224) and save as .npz per video.")
    parser.add_argument('--frames_root', type=str, required=True, help='Root directory of resized frames (224x224)')
    parser.add_argument('--output_root', type=str, required=True, help='Output directory for CNN feature .npz files')
    parser.add_argument('--max_videos', type=int, default=76, help='Number of videos to process (default: 76)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CNN feature extraction')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    precompute_cnn_features(args.frames_root, args.output_root, args.max_videos, args.batch_size, device)
