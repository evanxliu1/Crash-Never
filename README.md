# Collision Prediction System (Nexar Dataset)

## Project Overview
This project aims to predict vehicle collisions from dashcam video using a modular deep learning pipeline. The design process emphasizes efficiency, clarity, and principled decision-making at each stage.

---

## Design Journey & Step-by-Step Plan

### 1. Frame Extraction *(Completed)*
- **Goal:** Extract the most informative frames while saving memory and computation.
- **Design Choice:** Only the **middle 50%** of frames are extracted from each video (skipping the first and last 25%) to avoid uninformative intro/outro segments and reduce data volume.
- **Status:** Done. Full-resolution frames (1280×720) are used for all detection and tracking steps.

    - *Note: Resizing to 224×224 is only planned for the optional CNN feature extraction experiment (not used for YOLO or SORT).*
### 2. YOLO Object Detection *(Completed)*
- **Goal:** Detect all relevant objects (vehicles, pedestrians, etc.) in each extracted frame.
- **Model:** YOLOv11
- **Status:** Done. Detections saved for all frames.

### 3. Optical Flow (Ablated)
- **Goal:** Capture object and camera motion between frames.
- **Design Outcome:** Prototyped optical flow pipeline, but found it **too slow and computationally expensive** for large-scale batch processing. Decision: **abandon in favor of SORT tracking**.

### 4. SORT Tracking *(Next Step)*
- **Goal:** Track detected objects across frames to obtain temporal trajectories, velocities, and accelerations.
- **Rationale:** SORT is fast, robust, and well-suited for real-time or large-scale video analysis. It provides per-object motion statistics critical for collision prediction.
- **Status:** To be implemented (`batch_sort_tracking.py`).

### 5. Feature Construction *(Upcoming Experiments)*
- **Goal:** Build a rich feature representation for each time window.
- **Plan:**
    - **YOLO + SORT features:** Use object detections and tracked motion statistics as the primary feature set.
    - **CNN features (optional):** Plan to experiment with adding global frame features (e.g., ResNet-18 activations) for scene context. Will compare performance:
        - YOLO + SORT features **vs.** YOLO + SORT + CNN features
- **Status:** CNN feature extraction not yet implemented; will be tested as an ablation.

### 6. Sequence Modeling *(Planned)*
- **Goal:** Predict collisions from sequences of feature vectors.
- **Model:** LSTM baseline (with potential to try transformers/TCN in future).
- **Plan:** Train on windowed feature sequences; evaluate using standard metrics.

### 7. Evaluation & Visualization *(Planned)*
- **Goal:**
    - Quantitatively evaluate model performance.
    - Overlay predictions and object tracks on video for interpretability.
- **Plan:** Update visualization scripts to show SORT trajectories and model predictions.

### 8. Documentation & Refactoring *(Ongoing)*
- **Goal:**
    - Keep the codebase modular and maintainable.
    - Remove obsolete scripts/data (especially optical flow), document all design decisions, and ensure all scripts use progress bars (tqdm).

---

## Summary Table: Progress & Next Steps

| Step                          | Status      | Key Decisions/Notes                                             |
|-------------------------------|-------------|----------------------------------------------------------------|
| Frame Extraction              | Done        | Middle 50% only; resized to 224×224                            |
| YOLO Object Detection         | Done        | YOLOv11 on all extracted frames                                |
| Optical Flow                  | Ablated     | Too slow; replaced by SORT                                     |
| SORT Tracking                 | Next        | Fast, robust; to be implemented                                |
| Feature Construction          | Upcoming    | Compare YOLO+SORT vs. YOLO+SORT+CNN features                   |
| Sequence Modeling (LSTM)      | Planned     | Baseline; may try transformers/TCN later                       |
| Evaluation & Visualization    | Planned     | Overlay tracks/predictions; quantitative metrics                |
| Documentation & Refactoring   | Ongoing     | Remove obsolete code/data; document design process              |

---

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

---

## Design Principles
- **Efficiency:** Only process the most informative frames; avoid expensive computations unless justified by results.
- **Modularity:** Each step is a separate, testable module/script.
- **Transparency:** All design choices and ablations are documented for reproducibility and discussion.
- **Experimentation:** Plan to compare feature sets and models to find the best approach.

---

## Contact
For questions or collaboration, please contact the project maintainer.
