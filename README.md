# Collision Prediction System (Nexar Dataset)

## Project Overview
This project aims to build a deep learning pipeline for predicting vehicle collisions using dashcam video from the Nexar dataset. The pipeline leverages object detection (YOLO), object tracking (SORT), CNN feature extraction, and sequence modeling (LSTM, with potential for transformer-based models in the future).

## Current Status
- **YOLO object detection** completed for all frames.
- **Frame extraction** and resizing to 224×224 completed.
- **CNN features** (ResNet-18) pre-computed for first 76 training videos.
- **Optical flow** prototyped but will be replaced by **SORT** tracking.
- **LSTM baseline** trained on windowed CNN features (to be updated with SORT features).
- **Project structure** reorganized for modularity and clarity.

## Repository Structure
```
CrashDetection/
├── src/                # Core modules (feature extraction, dataset, models)
├── scripts/            # Pipeline & utility scripts
├── data/               # Processed data (frames, detections, features)
├── models/             # Model checkpoints
├── original_data/      # Raw MP4 videos
├── notebooks/          # EDA, visualization (empty for now)
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and plan
```

## Pipeline Steps
1. **Frame Extraction**: Extract 30 FPS frames from raw MP4s and resize to 224×224.
2. **YOLO Detection**: Run YOLO to get object detections for each frame.
3. **CNN Feature Extraction**: Pre-compute ResNet-18 features for each frame.
4. **SORT Tracking** (in progress): Track objects across frames using SORT, derive per-object velocity/acceleration.
5. **Feature Aggregation**: Combine CNN, YOLO, and SORT features into windowed feature vectors.
6. **Sequence Modeling**: Train LSTM (or transformer) on windowed features to predict collisions.
7. **Evaluation & Visualization**: Overlay predictions and trajectories for analysis.

## Next Steps
- [ ] Implement batch SORT tracking on YOLO detections
- [ ] Design new feature schema (combine CNN, YOLO, SORT)
- [ ] Update dataset/loader for new schema
- [ ] Remove optical flow code/data
- [ ] Retrain sequence model with new features
- [ ] Add learning rate scheduling
- [ ] Update visualizations for SORT trajectories
- [ ] Document pipeline and results

## Design Decisions & Notes
- **Why LSTM?**: LSTMs are well-suited for temporal sequence modeling in video, but transformers (e.g., TimeSformer, ViViT) are potential future upgrades.
- **Feature Schema**: The final feature vector for each window will concatenate CNN, YOLO, and SORT-derived statistics.
- **Progress Bars**: All long-running scripts use tqdm for transparency.
- **Modularity**: Pipeline is split into scripts and core modules for clarity and reproducibility.

## Work in Progress
This repository is under active development. See the plan below for detailed progress and next steps. Contributions, suggestions, and feedback are welcome!

---

# Development Plan

## Task List
- [x] Dataset exploration & cleaning
- [x] 30 FPS frame extraction
- [x] Batch YOLO object detection
- [x] Resize frames to 224×224
- [x] Implement CNN feature extractor & pre-compute features
- [x] Implement sliding-window feature aggregation
- [x] Build PyTorch Dataset + LSTM baseline and run initial training
- [ ] Harden `compute_relative_velocity` against malformed bbox data *(may be dropped with SORT)*
- [ ] Add learning-rate scheduling to training script
- [ ] Replace optical flow with SORT tracking
    - [ ] Implement batch SORT tracking on YOLO detections
    - [ ] Derive per-frame motion statistics (object velocity, acceleration)
    - [ ] Design new feature schema (CNN + YOLO + SORT) and update combination script (`feature_combination_v3.py`)
    - [ ] Update Dataset classes to read new features
    - [ ] Remove optical-flow code paths & dependencies (delete flow scripts & data)
- [ ] Re-train sequence model with SORT features; evaluate & compare
- [ ] Update visualization pipeline to overlay SORT trajectories & predictions
- [ ] Document codebase, preprocessing pipeline, and experimental results

## Current Goal
Integrate SORT tracking & new feature schema.

## Contact
For questions or collaboration, please contact the project maintainer.
