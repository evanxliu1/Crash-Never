"""
visualize_predictions_and_tracks.py

Overlays model predictions and (optionally) object tracks on video frames for qualitative analysis and visualization.

- Input: Videos, predictions, (optional) tracks
- Output: Annotated video frames or video files
- Usage: python visualize_predictions_and_tracks.py --video_dir <videos> --predictions <preds.npy> [--tracks <tracks_dir>]
- Dependencies: opencv-python, numpy, tqdm
"""

import os
import cv2
import numpy as np
import torch
from lstm_sequence_model import LSTMSequenceModel

# --- CONFIG ---
VIDEO_ID = '00077'  # Change as needed
FRAME_DIR = f'extracted_frames/{VIDEO_ID}'  # e.g. extracted_frames/00003/
FEATURE_PATH = f'windowed_features/{VIDEO_ID}_windowed.npy'
MODEL_PATH = "best_lstm_model.pt"  # If you want to load weights, set path here
OUTPUT_VIDEO = f'prediction_overlay_{VIDEO_ID}.mp4'
FPS = 30
WINDOW_SIZE = 30
STRIDE = 6

# --- LOAD FEATURES AND MODEL ---
features = np.load(FEATURE_PATH)  # (num_windows, window_size, 525)
features_torch = torch.tensor(features, dtype=torch.float32)

model = LSTMSequenceModel()
if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    logits = model(features_torch)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of accident
    preds = (probs > 0.5).astype(np.uint8)

# --- LOAD FRAMES ---
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.png')])
num_frames = len(frame_files)

# --- Calculate valid frame range (skip first/last 25%) ---
start_idx = int(0.25 * num_frames)
end_idx = int(0.75 * num_frames)
used_frames = frame_files[start_idx:end_idx]

# --- Map window predictions to frames ---
frame_pred = np.zeros(len(frame_files), dtype=np.uint8)
for widx, pred in enumerate(preds):
    win_start = widx * STRIDE
    win_end = win_start + WINDOW_SIZE
    for fidx in range(win_start, min(win_end, len(frame_pred))):
        frame_pred[fidx] = max(frame_pred[fidx], pred)

# --- Write output video with overlay ---
H, W = None, None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

for i, fname in enumerate(frame_files):
    fpath = os.path.join(FRAME_DIR, fname)
    frame = cv2.imread(fpath)
    if frame is None:
        continue
    if H is None or W is None:
        H, W = frame.shape[:2]
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))
    # Overlay prediction
    if frame_pred[i]:
        cv2.putText(frame, 'RISK', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)
        cv2.rectangle(frame, (10,10), (W-10,H-10), (0,0,255), 8)
    else:
        cv2.putText(frame, 'SAFE', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
    out.write(frame)

if out is not None:
    out.release()
print(f'Output video saved to {OUTPUT_VIDEO}')
