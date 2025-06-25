"""
yolo_utils.py

Core YOLO model loading, inference, and result parsing utilities for use in detection scripts and pipelines.

Functions:
- load_yolo_model(...)
- run_yolo_on_image(...)
- parse_yolo_results(...)

Intended to be imported by scripts such as yolo_detect.py.
"""

import torch
import cv2
import numpy as np
from pathlib import Path

def load_yolo_model(model_path=None, version='yolov11'):
    """
    Loads a YOLOv11 model for detection.
    - model_path: Path to YOLO checkpoint (.pt). If None, defaults to './models/yolo11n.pt'.
    - version: Specify YOLO version (default 'yolov11' for this project).
    Returns: model object ready for inference.
    """
    if model_path is None:
        model_path = str(Path(__file__).parent.parent / 'models' / 'yolo11n.pt')
    model_path = str(model_path)
    # Try to load YOLOv11 using torch.hub or a custom repo
    try:
        # If using a custom YOLOv11 repo, update the repo string below
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    except Exception as e:
        raise RuntimeError(f'Could not load YOLOv11 model from {model_path}. Ensure the repo and weights exist. Error: {e}')
    return model

def run_yolo_on_image(model, image):
    # image: numpy array (H, W, C) in BGR
    # model expects RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    return results

def parse_yolo_results(results):
    # Returns a list of dicts: [{'bbox': [x1, y1, x2, y2], 'conf': float, 'class': int, 'class_name': str}, ...]
    parsed = []
    if hasattr(results, 'pandas'):
        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            parsed.append({
                'bbox': [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])],
                'conf': float(row['confidence']),
                'class': int(row['class']),
                'class_name': str(row['name'])
            })
    else:
        # fallback, if results is a list/array
        for det in results:
            parsed.append(det)
    return parsed
