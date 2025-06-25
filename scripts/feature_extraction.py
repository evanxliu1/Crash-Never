import os
import json
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO  # Supports YOLOv11 and earlier

def run_yolo_on_frames(frames: List[np.ndarray], model_path: Optional[str] = None, device: str = 'cpu') -> List[List[Dict[str, Any]]]:
    """
    Run YOLOv11 object detection on a list of frames.
    Returns a list of detections per frame.
    Each detection is a dict: {class_id, class_name, conf, bbox=[x1, y1, x2, y2]}
    """
    if model_path:
        model = YOLO(model_path)
    else:
        model = YOLO('yolo11n.pt')  # Use YOLOv11 nano by default for speed
    model.to(device)
    results = []
    for frame in frames:
        # YOLO expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = model(rgb, verbose=False)[0]
        frame_dets = []
        for box in pred.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            class_name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
            frame_dets.append({
                'class_id': cls_id,
                'class_name': class_name,
                'conf': conf,
                'bbox': xyxy
            })
        results.append(frame_dets)
    return results

def save_detections_json(detections: List[List[Dict[str, Any]]], output_dir: str):
    """
    Save detection results as per-frame JSON files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame_dets in enumerate(detections):
        out_path = os.path.join(output_dir, f"frame_{idx:05d}.json")
        with open(out_path, 'w') as f:
            json.dump(frame_dets, f, indent=2)

def extract_and_detect(
    video_path: str,
    frames_output_dir: Optional[str] = None,
    dets_output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    augment: bool = False,
    yolo_model_path: Optional[str] = None,
    device: str = 'cpu'
):
    """
    Extract frames and run YOLO detection, saving results to disk.
    """
    
    aug_fn = example_augmentation if augment else None
    frames = extract_frames(
        video_path,
        output_dir=frames_output_dir,
        fps=30,
        resize=(1280, 720),
        augment_fn=aug_fn,
        max_frames=max_frames
    )
    detections = run_yolo_on_frames(frames, model_path=yolo_model_path, device=device)
    if dets_output_dir:
        save_detections_json(detections, dets_output_dir)
    print(f"Processed {len(frames)} frames from {video_path}.")
    if dets_output_dir:
        print(f"Detections saved to {dets_output_dir}")
    return detections

import tempfile
import shutil
import subprocess

def read_flo(file_path):
    import numpy as np
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        return np.resize(data, (h, w, 2))

def compute_optical_flow(frames, flow_output_dir=None, stats_output_path=None, nvof_exe_path=None):
    """
    Compute dense optical flow between consecutive frames using NVIDIA Optical Flow SDK sample app.
    Optionally save flow magnitude/angle arrays and summary stats (mean, std) per frame-pair.
    Args:
        frames: list of np.ndarray (BGR)
        flow_output_dir: directory to save .npz files of flow magnitude/angle
        stats_output_path: path to save summary stats (JSON)
        nvof_exe_path: path to NVIDIA Optical Flow sample executable
    Returns:
        flow_stats: list of dicts with mean/std for magnitude and angle per frame-pair
    """
    import shutil
    if nvof_exe_path is None:
        nvof_exe_path = os.environ.get('NVOF_EXE_PATH', r"C:\SDKS\OpticalFlowSDK\Optical_Flow_SDK_5.0.7\NvOFBasicSamples\build\AppOFCuda\Release\AppOFCuda.exe")
    os.makedirs(flow_output_dir, exist_ok=True) if flow_output_dir else None
    flow_stats = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(1, len(frames)):
            f0 = frames[i-1]
            f1 = frames[i]
            # Check frame validity
            def frame_ok(f):
                return (f is not None and isinstance(f, np.ndarray) and f.dtype == np.uint8 and len(f.shape) == 3 and f.shape[2] == 3 and f.shape[0] > 0 and f.shape[1] > 0)
            if not frame_ok(f0) or not frame_ok(f1):
                print(f"[ERROR] Invalid frame(s) at indices {i-1}, {i}. Shapes: {getattr(f0, 'shape', None)}, {getattr(f1, 'shape', None)} Types: {type(f0)}, {type(f1)}")
                continue
            f0_path = os.path.join(temp_dir, f'frame_{i-1:05d}.jpg')
            f1_path = os.path.join(temp_dir, f'frame_{i:05d}.jpg')
            out_flo = os.path.join(temp_dir, f'flow_{i-1:05d}_{i:05d}.flo')
            cv2.imwrite(f0_path, f0, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(f1_path, f1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # Check files exist
            if not (os.path.exists(f0_path) and os.path.exists(f1_path)):
                print(f"[ERROR] Frame image files not written for pair {i-1}-{i}.")
                continue
            cmd = [nvof_exe_path, '--input0', f0_path, '--input1', f1_path, '--output', out_flo]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True)
                if not os.path.exists(out_flo):
                    print(f"[ERROR] Optical Flow .flo file not written for pair {i-1}-{i}.")
                    continue
                flow = read_flo(out_flo)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                stats = {
                    'pair': [i-1, i],
                    'mag_mean': float(np.mean(mag)),
                    'mag_std': float(np.std(mag)),
                    'ang_mean': float(np.mean(ang)),
                    'ang_std': float(np.std(ang))
                }
                flow_stats.append(stats)
                if flow_output_dir:
                    np.savez_compressed(os.path.join(flow_output_dir, f'flow_{i-1:05d}_{i:05d}.npz'), mag=mag, ang=ang)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] NVIDIA Optical Flow failed for pair {i-1}-{i}: {e}")
                print(f"  Command: {' '.join(cmd)}")
                print(f"  Frame0 shape: {f0.shape}, dtype: {f0.dtype} | Frame1 shape: {f1.shape}, dtype: {f1.dtype}")
                print(f"  stderr: {e.stderr.decode(errors='ignore') if e.stderr else 'None'}")
                # Optionally, copy frames to persistent dir for inspection
                debug_dir = os.path.join(os.getcwd(), 'flow_debug')
                os.makedirs(debug_dir, exist_ok=True)
                shutil.copy(f0_path, os.path.join(debug_dir, f'debug_{i-1:05d}.jpg'))
                shutil.copy(f1_path, os.path.join(debug_dir, f'debug_{i:05d}.jpg'))
                continue
            except Exception as e:
                print(f"[ERROR] Unexpected error for pair {i-1}-{i}: {e}")
                continue
    if stats_output_path:
        with open(stats_output_path, 'w') as f:
            json.dump(flow_stats, f, indent=2)
    return flow_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames, run YOLOv11 detection, and compute optical flow on a video.")
    parser.add_argument("video_path", type=str, nargs="?", default=None, help="Path to the .mp4 video file")
    parser.add_argument("--frames_output_dir", type=str, default=None, help="Directory to save extracted frames")
    parser.add_argument("--frames_input_dir", type=str, default=None, help="Directory to load pre-extracted frames from (overrides video extraction)")
    parser.add_argument("--dets_output_dir", type=str, default=None, help="Directory to save detection JSONs")
    parser.add_argument("--flow_output_dir", type=str, default=None, help="Directory to save optical flow .npz files")
    parser.add_argument("--flow_stats_path", type=str, default=None, help="Path to save flow summary stats JSON")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--augment", action="store_true", help="Apply example augmentation")
    parser.add_argument("--yolo_model_path", type=str, default=None, help="Path to custom YOLOv11 model (optional)")
    parser.add_argument("--device", type=str, default='cpu', help="Device for YOLO inference (cpu or cuda)")
    args = parser.parse_args();

    # Load frames
    from video_utils import example_augmentation
    aug_fn = example_augmentation if args.augment else None
    if args.frames_input_dir:
        # Load frames from directory
        import cv2
        import numpy as np
        frame_files = sorted([f for f in os.listdir(args.frames_input_dir) if f.lower().endswith(('.jpg','.png'))])
        frames = [cv2.imread(os.path.join(args.frames_input_dir, f)) for f in frame_files]
        if args.max_frames:
            frames = frames[:args.max_frames]
        print(f"Loaded {len(frames)} frames from {args.frames_input_dir}")
    elif args.video_path:
        frames = extract_frames(
            args.video_path,
            output_dir=args.frames_output_dir,
            fps=30,
            resize=(1280, 720),
            augment_fn=aug_fn,
            max_frames=args.max_frames
        )
    else:
        raise ValueError("Either --frames_input_dir or video_path must be provided.")
    # YOLO detection
    detections = run_yolo_on_frames(frames, model_path=args.yolo_model_path, device=args.device)
    if args.dets_output_dir:
        save_detections_json(detections, args.dets_output_dir)
    print(f"Processed {len(frames)} frames from {args.video_path}.")
    if args.dets_output_dir:
        print(f"Detections saved to {args.dets_output_dir}")
    # Optical flow
    if args.flow_output_dir or args.flow_stats_path:
        flow_stats = compute_optical_flow(frames, flow_output_dir=args.flow_output_dir, stats_output_path=args.flow_stats_path)
        print(f"Optical flow computed for {len(flow_stats)} frame pairs.")
        if args.flow_stats_path:
            print(f"Flow stats saved to {args.flow_stats_path}")
